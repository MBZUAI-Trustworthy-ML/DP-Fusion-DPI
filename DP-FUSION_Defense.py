import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
import numpy as np
from bisect import bisect_left, bisect_right
from torch.nn.utils.rnn import pad_sequence
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
from copy import deepcopy
import torch.nn.functional as F
import pickle
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from math import log, exp
import math
from typing import Dict, List, Union,Sequence, Tuple, Any
from tqdm import tqdm
import time


DEFAULT_BETA_DICT = {
    "PERSON":   0.5,
    "CODE":     0.5,
    "LOC":      0.5,
    "ORG":      0.5,
    "DEM":      0.5,
    "DATETIME": 0.5,
    "QUANTITY": 0.5,
    "MISC":     0.5,
}

# Command line argument parsing
parser = argparse.ArgumentParser(description="DP-Fusion implementation with multi-GPU support")
parser.add_argument("--input_file", type=str, required=True, help="what file to use for input")
parser.add_argument("--start", type=int, required=True, help="Start index for data processing")
parser.add_argument("--end", type=int, required=True, help="End index for data processing")
parser.add_argument("--gpu", type=str, required=True, help="Comma-separated list of GPU indices to use")
parser.add_argument("--output_dir", type=str, default="output/", help="Directory for output files")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID to use")
parser.add_argument("--output_file", type=str, default="output.json", help="Output file name")
parser.add_argument("--output_file_pickle", type=str, default="output.json", help="Output file name")
parser.add_argument("--output_file_meta_data", type=str, default="output.json", help="Output file name")
parser.add_argument("--alpha", type=float, default=2.0, help="Default alpha value for all entities")
parser.add_argument("--to_cache", type=bool, default=True, help="do u want caching this will feast on VRAM like its cheescake")
parser.add_argument("--to_debug", type=bool, default=False, help="do u want to debug this code, enable for many prints")
parser.add_argument("--independent", type=bool, default=False, help="do u want to run the independent grp version of the code")
parser.add_argument("--hf_token", type=str, required=True, help="your hf token")
parser.add_argument("--max_tokens", type=int, default=900, help="max tokens to generate")

parser.add_argument(
        "--beta_dict",
        type=json.loads,
        default=json.dumps(DEFAULT_BETA_DICT),
        help=(
            "JSON dict of per‑group β's (e.g. "
            "'{\"PERSON\":0.2,\"CODE\":0.1,…}')"
        ),
    )

args = parser.parse_args()

INPUT_FILE = args.input_file
TO_CACHE = args.to_cache
DEBUG_MODE = args.to_debug

# print("Running with base variables: ", args)

# Remove test file if it exists (from the original code)

# Configure GPU settings
available_gpus = [int(gpu) for gpu in args.gpu.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Optimize CPU threading
num_threads = min(1, os.cpu_count() or 1)
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

# Constants
MAX_TOKENS_GLOBAL = args.max_tokens  # Maximum number of tokens to generate I AM SETTING THIS TO AVG SIZE OF PUBLIC TEXT INPUT
PLACEHOLDER_TOKEN = "_"  # Placeholder for redacted content
HF_TOKEN = args.hf_token  # Hugging Face token

# Entity types and lambda values
ENTITY_TYPES = [
    "PERSON", "CODE", "LOC", "ORG", "DEM", 
    "DATETIME", "QUANTITY", "MISC"
]

BETA_DICT = args.beta_dict

ALPHA = args.alpha

print("Recieved Lambda Dict: ", BETA_DICT)


# Configure logging
def log(message, level="INFO"):
    """Simple logging function with timestamp and log level."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# ===================================
# TOKEN REPLACEMENT AND PROCESSING
# ===================================

def replace_sequences_with_placeholder_fast(text, word_offsets, placeholder, tokenizer):
    """
    Replace tokens falling within provided word offset ranges with placeholder tokens.
    
    Args:
        text (str): Original text string
        word_offsets (list): List of [start_char, end_char] offsets for words to replace
        placeholder (str): Placeholder token to use (e.g., "_")
        tokenizer: Tokenizer that returns 'input_ids' and 'offset_mapping'
    
    Returns:
        list: Token IDs with specified words replaced by placeholder token ID
    """
    # Get placeholder token ID
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    
    # Encode text to get token offsets with cache
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    
    # Sort word_offsets by start position for efficient lookup
    word_offsets = sorted(word_offsets, key=lambda x: x[0])
    starts = [wo[0] for wo in word_offsets]
    ends = [wo[1] for wo in word_offsets]
    
    # Process each token to check for overlaps with word offsets
    for i, (t_start, t_end) in enumerate(offsets):
        if t_start == t_end:
            # Skip empty offsets
            continue
            
        # Find position to insert t_end in starts
        idx = bisect_right(starts, t_end)
        
        # Check for overlaps with words that start before t_end
        while idx > 0:
            idx -= 1
            w_start, w_end = starts[idx], ends[idx]
            
            # Check overlap condition: w_end > t_start and w_start < t_end
            if w_end > t_start and w_start < t_end:
                # Overlapping word found, replace token
                input_ids[i] = placeholder_id
                break
    
    return input_ids


# ===================================
# MODEL UTILITIES
# ===================================

def create_redacted_versions(text, grouped_offsets, placeholder_token, tokenizer, entity_types_list):
    """
    Produces multiple versions of the input text:
    - 'PUBLIC': All entities replaced by placeholder
    - One per entity type: everything replaced EXCEPT that type
    
    Args:
        text (str): Original text string
        grouped_offsets (dict): Dictionary mapping entity types to lists of character offsets
        placeholder_token (str): Token to use as placeholder
        tokenizer: Tokenizer to use
        entity_types_list (list): List of entity types
        
    Returns:
        tuple: (versions dict, token_ids dict)
    """
    versions = {}
    token_ids_out = {}
    
    # 1. PUBLIC version: Replace ALL entities
    all_offsets = []
    for off_list in grouped_offsets.values():
        all_offsets.extend(off_list)
        
    public_ids = replace_sequences_with_placeholder_fast(text, all_offsets, placeholder_token, tokenizer)
    token_ids_out["PUBLIC"] = public_ids
    versions["PUBLIC"] = tokenizer.decode(public_ids)
    
    # 2. One version for each entity type (everything replaced EXCEPT that type)
    # SOLVED ONE FOR EACH TYPE PRESENT ONLY!

    for ent_type in grouped_offsets:
        offsets_except_this_type = []
        for t2, off_list in grouped_offsets.items():
            if t2 != ent_type:
                offsets_except_this_type.extend(off_list)
                
        ids_except_this_type = replace_sequences_with_placeholder_fast(
            text, offsets_except_this_type, placeholder_token, tokenizer
        )
        versions[ent_type] = tokenizer.decode(ids_except_this_type)
        token_ids_out[ent_type] = ids_except_this_type
        
    return versions, token_ids_out

def compute_renyi_divergence_clipped_symmetric(p, q, alpha, eps: float = 1e-10):
    """
    Symmetric Rényi divergence  D↔_α (p‖q) = max{ D_α(p‖q), D_α(q‖p) }.

    Args
    ----
    p, q : torch.Tensor
        Probability vectors (last dimension is the support).
    alpha : float  (must be > 1)
    eps   : float  (small constant for numerical stability)

    Returns
    -------
    torch.Tensor
        D↔_α (p, q)  with shape p.shape[:-1]
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")

    # promote to float32 and clamp to avoid log/0
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)

    # forward direction  D_α(p‖q)
    term_pq = torch.sum(p.pow(alpha) * q.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_pq  = (1.0 / (alpha - 1.0)) * torch.log(term_pq)

    # reverse direction  D_α(q‖p)
    term_qp = torch.sum(q.pow(alpha) * p.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_qp  = (1.0 / (alpha - 1.0)) * torch.log(term_qp)

    # symmetric divergence
    return torch.maximum(div_pq, div_qp)

def find_lambda(p_priv, p_pub, alpha, beta, debug_mode, max_iter=20, tol=1e-6):
    """
    Binary search for λ in [0,1] that satisfies:
        D_alpha( (1-λ)*p_pub + λ*p_priv  || p_pub ) <= beta

    Returns:
        (lambda_value, divergence)

    p_priv, p_pub: Probability distributions (already softmaxed & temperature-scaled).
    alpha: float > 1
    beta:  float >= 0
    """
    if beta <= 0:
        # If beta == 0, the mixture must match p_pub exactly => lambda=0.
        return 0.0, 0.0

    # Check divergence at λ=1 (fully private). If even that is <= beta, we can use full private.
    div_at_1 = compute_renyi_divergence_clipped_symmetric(p_priv, p_pub, alpha)
    # if debug_mode:
    #     print(f"Renyi divergence at λ=1: {div_at_1:.4f}")

    if div_at_1 <= beta:
        return 1.0, div_at_1

    # Otherwise, perform a bisection search in [0,1].
    left, right = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        # Compute the mixture
        mixture = mid * p_priv + (1 - mid) * p_pub
        div = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)

        # if debug_mode:
        #     print(f"λ={mid:.4f}, divergence={div:.4f}, left={left:.4f}, right={right:.4f}")

        if div > beta:
            right = mid
        else:
            left = mid

        if (right - left) < tol:
            break

    # Recompute the divergence using the final λ (left) to ensure accuracy.
    final_lambda = left
    mixture = final_lambda * p_priv + (1 - final_lambda) * p_pub
    final_div = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)
    return final_lambda, final_div

def compute_dp_epsilon(
    divergences: Dict[str, List[float]],
    alpha: float,
    delta: float,
    mode: str = "global"
) -> Union[float, Dict[str, float]]:
    """
    Compute (ε, δ)-DP guarantee from per-step Rényi divergences, with debug prints.

    Args:
        divergences: mapping group_name -> list of β_t values (length=T).
                     The key "PUBLIC" (if present) will be ignored.
        alpha: Rényi order (>1).
        delta: target δ in (ε, δ)-DP.
        mode: "global" for one ε protecting all groups (worst-case per step),
              "per_group" for a dict of ε_i per group.

    Returns:
        If mode == "global": float ε.
        If mode == "per_group": dict of {group: ε_i}.
    """
    print(f"[DEBUG] Starting compute_dp_epsilon with alpha={alpha}, delta={delta}, mode='{mode}'")
    # 1) Basic checks
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0,1)")

    # 2) Filter out PUBLIC and ensure at least one private group
    priv_div = {g: lst for g, lst in divergences.items() if g != "PUBLIC"}
    print(f"[DEBUG] Private groups found: {list(priv_div.keys())}")
    if not priv_div:
        raise ValueError("No private groups provided")

    # 3) Ensure all groups have same number of steps
    step_counts = {len(lst) for lst in priv_div.values()}
    print(f"[DEBUG] Divergence list lengths per group: {step_counts}")
    if len(step_counts) != 1:
        raise ValueError(f"Divergence lists have unequal lengths: {step_counts}")
    T = step_counts.pop()
    N = len(priv_div)
    print(f"[DEBUG] Number of groups N={N}, number of tokens T={T}")

    # 4) Define per-step RDP cost
    def eps_step(beta: float, step: int = None) -> float:
        if beta is None:
            raise ValueError("Found None in divergence list")
        # Argument of log must be positive
        arg = (N - 1.0)/N + (1.0/N)*math.exp((alpha - 1.0)*4.0*beta)
        if arg <= 0.0:
            raise ValueError(f"Non-positive argument for log: {arg}")
        eps = (1.0/(alpha - 1.0)) * math.log(arg)
        # if step is not None:
        #     print(f"[DEBUG] Step {step}: beta={beta:.6f}, log-arg={arg:.6f}, eps_step={eps:.6f}")
        return eps

    # 5) Compute total RDP and convert to DP
    if mode == "global":
        total_rdp = 0.0
        for t in range(T):
            # Collect betas at step t
            betas = [div_list[t] for div_list in priv_div.values()]
            print(f"[DEBUG] Global mode, step {t}: betas={betas}")
            beta_max = max(betas)
            total_rdp += eps_step(beta_max, step=t)
        epsilon = total_rdp + math.log(1.0/delta)/(alpha - 1.0)
        print(f"[DEBUG] Total RDP={total_rdp:.6f}, final ε={epsilon:.6f}")
        return epsilon

    elif mode == "per_group":
        epsilons = {}
        for group, div_list in priv_div.items():
            print(f"[DEBUG] ---- Group '{group}' ----")
            print(f"[DEBUG] Divergences (length {len(div_list)}): {div_list[:5]}{'...' if len(div_list)>5 else ''}")
            total_rdp_g = 0.0
            for t, beta_t in enumerate(div_list):
                total_rdp_g += eps_step(beta_t, step=t)
            eps_group = total_rdp_g + math.log(1.0/delta)/(alpha - 1.0)
            print(f"[DEBUG] Group '{group}': Total RDP={total_rdp_g:.6f}, ε={eps_group:.6f}")
            epsilons[group] = eps_group
        return epsilons

    else:
        raise ValueError("mode must be 'global' or 'per_group'")

def half_past(past):
    return tuple(tuple(kv.to(dtype=torch.float16) for kv in layer) for layer in past)

def dp_fusion_groups_incremental(
    token_ids_groups, beta_dict, alpha, model, tokenizer,
    temperature=1.0, max_new_tokens=50, debug_mode=DEBUG_MODE,
    device_map=None, batch_override=None
):
    """
    DP-Fusion generation with true incremental decoding (using caching) and
    multi-group support. In this variant:
      1. Each group's full prefix (a 1D tensor of token IDs) is processed in one batch,
         forming the initial cache.
      2. Then, at each generation step, only the new token is passed (shape: [num_groups, 1])
         along with past_key_values.
    
    Although the same next token is sampled for every group (for fusion),
    the cache (past_key_values) preserves each group's distinct context.
    
    Args:
        token_ids_groups (dict): 
          - Keys: group names ("PUBLIC", "high", "med", "low", etc.)
          - Values: list or 1D torch.Tensor of token IDs representing the group's prefix.
        beta_dict (dict): Mapping from group name to β threshold.
        alpha (float): Rényi divergence order (>1).
        model: Hugging Face CausalLM.
        tokenizer: Corresponding tokenizer.
        temperature (float): Temperature for scaling logits.
        max_new_tokens (int): Maximum tokens to generate.
        debug_mode (bool): Whether to print debug information.
        device_map (dict): Optional device map.
        batch_override (dict): Optional batch settings override.
    
    Returns:
        str: Final generated text from the "PUBLIC" group.
    """

    eos_id = tokenizer.eos_token_id

    going_lambdas = {}
    going_divergence = {}

    # Validate inputs
    if "PUBLIC" not in token_ids_groups:
        raise ValueError("Must have a 'PUBLIC' key in token_ids_groups.")
    private_groups = [g for g in token_ids_groups if g != "PUBLIC"]
    if not private_groups:
        raise ValueError("No private groups besides 'PUBLIC' – need at least one for DP-Fusion.")
    
    # Determine device
    if device_map:
        first_device = next(iter(device_map.values()))
        device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    else:
        device = model.device
    
    # Ensure all token sequences are 1D tensors on device.
    for group, tokens in token_ids_groups.items():
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        # Expecting token_ids_groups[group] shape: (L,)
        token_ids_groups[group] = tokens.to(device)
    
    if debug_mode:
        print(f"[DP-Fusion] Starting generation. Private groups: {private_groups}")
        for g in token_ids_groups:
            print(f"[Initial] Prefix shape for group {g}: {token_ids_groups[g].shape}")
    
    group_order = list(token_ids_groups.keys())
    num_groups = len(group_order)
    
    # === INITIAL PASS: Process each group's full prefix to build the cache ===
    # Instead of unsqueezing, we leave each prefix as a 1D tensor.
    prefix_batches = [token_ids_groups[g] for g in group_order]  # each: shape (L,)
    # pad_sequence expects a list of 1D tensors and returns shape (num_groups, max_length)
    input_batch = torch.nn.utils.rnn.pad_sequence(prefix_batches, batch_first=True, padding_value=tokenizer.pad_token_id)
    if debug_mode:
        print(f"[Initial] Input batch shape: {input_batch.shape}")
    
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        outputs = model(input_ids=input_batch, use_cache=True, past_key_values=None)
    #past = outputs.past_key_values  # This cache now holds each group's context.
    past = half_past(outputs.past_key_values)
    # Extract logits for the last token in each sequence.
    last_logits = outputs.logits[:, input_batch.size(1) - 1, :]  # shape: (num_groups, vocab_size)
    group_logits = {g: last_logits[i] for i, g in enumerate(group_order)}
    
    # Compute distributions per group (apply temperature scaling & softmax).
    pub_scaled = group_logits["PUBLIC"] / temperature
    p_pub = F.softmax(pub_scaled, dim=-1)
    p_priv_dict = {}
    for pg in private_groups:
        priv_scaled = group_logits[pg] / temperature
        p_priv_dict[pg] = F.softmax(priv_scaled, dim=-1)
    
    # DP-Fusion: Find lambdas and form the fused distribution.
    lambdas = {}
    for pg in private_groups:
        beta_val = beta_dict.get(pg)
        lam_pg, got_div = find_lambda(p_priv_dict[pg], p_pub, alpha, beta_val, debug_mode=debug_mode)
        lambdas[pg] = lam_pg
        if debug_mode:
            print(f"[Initial] Selected Lambda for group {pg}: {lam_pg}")
    
    sum_out = torch.zeros_like(p_pub)
    for pg in private_groups:
        lam_g = lambdas[pg]
        mix_g = lam_g * p_priv_dict[pg] + (1 - lam_g) * p_pub
        sum_out += mix_g
    p_out_avg = sum_out / len(private_groups)
    
    # Sample the first new token.
    next_token = torch.multinomial(p_out_avg, 1).item()
    if debug_mode:
        token_str = tokenizer.decode([next_token])
        print(f"[Initial] Sampled token '{token_str}' (ID={next_token})")
    
    # Append the new token to each group's prefix.
    for g in group_order:
        token_ids_groups[g] = torch.cat([token_ids_groups[g], torch.tensor([next_token], device=device)], dim=0)
    
    # === INCREMENTAL LOOP: Process only the new token each step ===
    for step in range(1, max_new_tokens):
        # Create new input: a tensor of shape (num_groups, 1) where each row is the new token.
        new_tokens_batch = torch.tensor([[next_token]] * num_groups, device=device)
        
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids=new_tokens_batch, past_key_values=past, use_cache=True)
        past = outputs.past_key_values  # Update the cache.
        # Extract logits for the new token (shape: (num_groups, vocab_size)).
        last_logits = outputs.logits[:, -1, :]
        group_logits = {g: last_logits[i] for i, g in enumerate(group_order)}
        
        # Compute distributions.
        pub_scaled = group_logits["PUBLIC"] / temperature
        p_pub = F.softmax(pub_scaled, dim=-1)
        p_priv_dict = {}
        for pg in private_groups:
            priv_scaled = group_logits[pg] / temperature
            p_priv_dict[pg] = F.softmax(priv_scaled, dim=-1)
        
        # DP-Fusion: find lambdas and combine distributions.
        lambdas = {}
        for pg in private_groups:
            beta_val = beta_dict.get(pg)
            lam_pg, div_got = find_lambda(p_priv_dict[pg], p_pub, alpha, beta_val, debug_mode=debug_mode)
            lambdas[pg] = lam_pg
            if debug_mode:
                print(f"[Step {step}] Selected Lambda for group {pg}: {lam_pg}")
            if pg not in going_lambdas:
                going_lambdas[pg] = []
                going_divergence[pg] = []
            going_lambdas[pg].append(lam_pg)
            going_divergence[pg].append(div_got)

        sum_out = torch.zeros_like(p_pub)
        for pg in private_groups:
        #     if debug_mode:
        #         print(f"Taking into account {pg} with lambda {lambdas[pg]} to mix")

                # l,d,is_mono = check_monotonicity(p_pub, p_priv_dict[pg])
                # print(f"Is monotonic: {is_mono}")


            mix_g = lambdas[pg] * p_priv_dict[pg] + (1 - lambdas[pg]) * p_pub
            sum_out += mix_g
        p_out_avg = sum_out / len(private_groups)



        
        # if debug_mode:
        #     top10 = torch.topk(p_out_avg, 10)
        #     top10_tokens = tokenizer.convert_ids_to_tokens(top10.indices.cpu().numpy())
        #     top10_probs = top10.values.cpu().numpy()
        #     print(f"[Step {step}] Top 10 tokens from fused distribution:")
        #     for i, (tok, prob) in enumerate(zip(top10_tokens, top10_probs)):
        #         print(f"[Step {step}] Token {i+1}: {tok} (prob={prob:.4f})")
        
        # Sample the next token.
        next_token = torch.multinomial(p_out_avg, 1).item()
        token_str = tokenizer.decode([next_token])
        # print(f"[Step {step}] Sampled token '{token_str}' (ID={next_token})")
        
        # Append the new token to each group's sequence.
        for g in group_order:
            token_ids_groups[g] = torch.cat([token_ids_groups[g], torch.tensor([next_token], device=device)], dim=0)
    
        # ======= END‑OF‑STEP CLEAN‑UP =======
        del outputs, last_logits, group_logits        # large, no longer needed
        torch.cuda.empty_cache()                      # give VRAM back to the allocator
        # ====================================

        if next_token == eos_id:
            break

    # Return the final text from the PUBLIC group.
    final_text = tokenizer.decode(token_ids_groups["PUBLIC"], skip_special_tokens=True)
    if debug_mode:
        print("[DP-Fusion] Generation complete.")

    torch.cuda.empty_cache()

    return final_text, going_lambdas, going_divergence

# ===================================
# PROMPT FORMATTING
# ===================================

def format_prompt_new_template(tokenizer, private_doc, placeholder):
    """
    Create a model-agnostic prompt for paraphrasing without placeholder markers.
    
    Args:
        tokenizer: Hugging Face tokenizer
        private_doc (str): Document to paraphrase (may contain underscores)
        
    Returns:
        str: Formatted prompt for the model
    """
    return(f"""<|im_start|>system
You are given a passage that may contain placeholders (underscores) or incomplete data. Your job is to produce a natural paraphrase. Do not use any underscores or placeholders in your output. If data is missing, just omit it or paraphrase gracefully. Do not output anything except the paraphrase. Make sure retain all information from the source document.<|im_end|>
<|im_start|>user
Document:\n\n{private_doc}
Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, as {placeholder} indicates redacted text. To ensure the generated text is as a natural as possible, you must never output the {placeholder} themselves. <|im_end|>
<|im_start|>assistant
Sure. Here is the paraphrased document without underscores or placeholders:""")

    """
    Scan divergences (group -> list[beta_t]) for None values.
    Raises a ValueError listing all (group, step) positions with None.
    """
    none_locs = [
        (group, idx)
        for group, div_list in divergences.items()
        for idx, beta in enumerate(div_list)
        if beta is None
    ]
    if none_locs:
        details = "; ".join(f"{g}@{i}" for g, i in none_locs)
        raise ValueError(f"None found in divergences at: {details}")
    # else everything is fine
# ===================================
# MAIN PROCESSING FUNCTION
# ===================================

def process_entry(entry, model, tokenizer, beta_dict, alpha, gpu_id=0,device_map=None):
    """
    Process a single entry with DP-Fusion.
    
    Args:
        entry (dict): Dictionary containing passage and entity data
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        lambda_dict (dict): Lambda values for each entity type
        gpu_id (int): GPU to use for this entry
        
    Returns:
        dict: Updated entry with redacted versions and generated text
    """
    output_entry = {}

    meta_save = {}

    passage_lines = entry["passage"]
    private_entities_all = entry["private_entities"]
    
    # Combine all lines into a single passage with offsets adjustment

    MAX_CHARS = 10000

    passage_new = ""
    offsets_new = {}
    going_offset = 0

    for i, line_text in enumerate(passage_lines):
        line_len = len(line_text) + 1   # +1 for the space/newline we add

        # If adding this line would exceed MAX_CHARS, stop entirely
        if going_offset + line_len > MAX_CHARS:
            break

        # Otherwise, record all its entity offsets (as before)
        entity_info_list = private_entities_all[i] if i < len(private_entities_all) else []
        for entity_info in entity_info_list:
            s, e = entity_info["offset"]
            e_type = entity_info["type"]
            offsets_new.setdefault(e_type, []).append((s + going_offset, e + going_offset))

        # Append the line
        passage_new += line_text + " "
        going_offset += line_len

    MAX_TOKENS = min(tokenizer(passage_new, return_tensors="pt")["input_ids"].shape[1],MAX_TOKENS_GLOBAL)

    # Create prompt for the model
    prompt = format_prompt_new_template(tokenizer, passage_new, placeholder=
                                        PLACEHOLDER_TOKEN)
    
    # Find where the actual document starts in the prompt
    document_marker = "Document:\n\n"
    document_start_index = prompt.find(document_marker)
    if document_start_index == -1:
        raise ValueError("Cannot find 'Document:\n\n' in the prompt.")
        
    # Calculate prefix length
    prefix_length = document_start_index + len(document_marker)
    
    # Adjust offsets to account for the prefix
    for key in offsets_new.keys():
        offsets_new[key] = [
            (start + prefix_length, end + prefix_length)
            for start, end in offsets_new[key]
        ]
    
    # Create redacted versions
    versions_dict, token_ids_dict = create_redacted_versions(
        prompt,
        offsets_new,
        PLACEHOLDER_TOKEN,
        tokenizer,
        ENTITY_TYPES
    )

    # print all versions
    for key, version in versions_dict.items():
        print(f"Version for {key}: {version}")
        print("=====================================")
    
    # Prepare token IDs for DP-Fusion
    token_ids_groups = {}
    for key, ids in token_ids_dict.items():
        token_ids_groups[key] = ids

    text,lambdas, divergences = dp_fusion_groups_incremental(
        token_ids_groups,
        beta_dict,
        alpha,
        model,
        tokenizer,
        temperature=1.0,
        max_new_tokens=MAX_TOKENS,
        debug_mode=DEBUG_MODE,    
            # Only show debug for first batch
    )

    epsilons = compute_dp_epsilon(divergences=divergences, alpha=2.0, delta=0.001, mode="per_group")

    # Properly printing the resultant epsilons
    for grp in epsilons:
        print(f"Epsilon for group {grp}: {epsilons[grp]}")

    print(f" text recieved is - {text}")
    print("=====================================")

    output_entry["redacted_text"] = text.split("Here is the paraphrased document without underscores or placeholders:")[1]
    return output_entry, meta_save
# ===================================
# if __name__ == "__main__":
#     main()
# ==================================
def main():
    start_time = time.time()
    log(f"Starting processing for data indices {args.start} to {args.end}")

    # Load input data
    try:
        with open(INPUT_FILE, "r") as json_file:
            big_data = json.load(json_file)
        data = big_data[args.start:args.end]
        log(f"Loaded {len(data)} entries from {INPUT_FILE} (indices {args.start} to {args.end}).")
    except Exception as e:
        log(f"Error loading input data: {str(e)}", "ERROR")
        return

    # Determine GPU distribution
    num_gpus = len(args.gpu.split(',')) if args.gpu.strip() else 0
    log(f"Using {num_gpus} GPUs: {args.gpu}")

    # Output file paths
    output_file = args.output_file
    output_file_pickle = args.output_file_pickle
    output_file_meta = args.output_file_meta_data

    # Resume from partial output if exists
    output_data = []
    meta_data_list = []

    already_processed = 0
    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                partial = json.load(f)
                output_data = partial
                already_processed = len(output_data)
            log(f"Resuming from {already_processed} already processed entries in {output_file}")
        except Exception as e:
            log(f"Error loading existing output file: {str(e)}", "ERROR")
            log("Starting from scratch (no partial resume).", "INFO")
            output_data = []
            already_processed = 0
    else:
        log("No existing output file found or file is empty, starting from scratch.")

    # Skip already processed entries
    if already_processed >= len(data):
        log("All entries have already been processed. Nothing to do.")
        return
    data_to_process = data[already_processed:]
    log(f"Processing {len(data_to_process)} new entries (from index {already_processed} to {len(data)-1}).")

    # Load model and tokenizer (unchanged except for logging)
    try:
        log(f"Loading model {args.model_id}")
        num_gpus = len(args.gpu.split(',')) if args.gpu.strip() else 0
        #device_map = "auto" if num_gpus > 1 else None
        device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_auth_token=HF_TOKEN
        )
        log("Model loaded successfully.")
    except Exception as e:
        log(f"Error loading model: {str(e)}", "ERROR")
        return

    # Process entries (resume from already_processed)
    for idx, entry in enumerate(tqdm(data_to_process, desc="Processing entries"), start=already_processed):
        # try:
        result, meta_data = process_entry(entry, model, tokenizer, BETA_DICT, args.alpha,device_map)
        
        output_data.append(result)
        meta_data_list.append(meta_data)

        # Save intermediate results periodically
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            with open(output_file_pickle, "wb") as f:
                pickle.dump(output_data, f)
            with open(output_file_meta, "wb") as f:
                pickle.dump(meta_data_list, f)
            
            log(f"Saved intermediate results after processing {idx + 1} entries.")
        except Exception as e:
            log(f"Error saving intermediate results: {str(e)}", "ERROR")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        with open(output_file_pickle, "wb") as f:
            pickle.dump(output_data, f)
        with open(output_file_meta, "wb") as f:
            pickle.dump(meta_data_list, f)
        log(f"Saved final results to {output_file} and {output_file_pickle}")
    except Exception as e:
        log(f"Error saving final results: {str(e)}", "ERROR")

    elapsed_time = time.time() - start_time
    log(f"Processing completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
