import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
import numpy as np
from bisect import bisect_left, bisect_right
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
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
import random
import copy

# ===================================
# CONFIGURATION AND SETUP
# ===================================

# Set matplotlib parameters for better visualization
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.family'] = 'DejaVu Sans'

COLORS = plt.get_cmap("tab10").colors

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

SEED = {
    "PERSON":   42,
    "CODE":     31,
    "LOC":      24,
    "ORG":      67,
    "DEM":      93,
    "DATETIME": 32,
    "QUANTITY": 36,
    "MISC":     91,    
}

ENTITIES_TO_ATTACK = ["CODE","PERSON","DATETIME"]


# Command line argument parsing
parser = argparse.ArgumentParser(description="DP-Fusion Attack implementation with multi-GPU support")
parser.add_argument("--input_file", type=str, required=True, help="what file to use for input")
parser.add_argument("--start", type=int, required=True, help="Start index for data processing")
parser.add_argument("--end", type=int, required=True, help="End index for data processing")
parser.add_argument("--gpu", type=str, required=True, help="Comma-separated list of GPU indices to use")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID to use")
parser.add_argument("--output_file", type=str, default="output.json", help="Output file name")
parser.add_argument("--attack_output_file_json", type=str, default="output.json", help="Output file name")
parser.add_argument("--attack_output_file_pickle", type=str, default="output.json", help="Output file name")
parser.add_argument("--attack_output_file_meta_data", type=str, default="output.json", help="Output file name")
parser.add_argument("--to_debug", type=bool, default=False, help="do u want to debug this code")
parser.add_argument("--number_of_cands", type=int, required=True, help="Candidates")
parser.add_argument("--hf_token", type=str, required=True, help="your hf token")

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
DEBUG_MODE = args.to_debug
OUTPUT_FILE = args.output_file
ATTACK_OUTPUT_FILE_PICKLE = args.attack_output_file_pickle
ATTACK_OUTPUT_FILE_JSON = args.attack_output_file_json
ATTACK_OUTPUT_FILE_META_DATA = args.attack_output_file_meta_data
CANDS = args.number_of_cands


# Remove test file if it exists (from the original code)

# Configure GPU settings
available_gpus = [int(gpu) for gpu in args.gpu.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Optimize CPU threading
num_threads = min(1, os.cpu_count() or 1)
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

# Constants 
PLACEHOLDER_TOKEN = "_"  # Placeholder for redacted content
HF_TOKEN = args.hf_token  # Hugging Face token


# Entity types and lambda values
ENTITY_TYPES = [
    "PERSON", "CODE", "LOC", "ORG", "DEM", 
    "DATETIME", "QUANTITY", "MISC"
]

# Configure logging
def log(message, level="INFO"):
    """Simple logging function with timestamp and log level."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


@torch.inference_mode()
def batched_logp_causal_single_pass_batched(
    prompts: List[str],
    paraphrase: str,
    model,
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 1.0,
    batch_size: int = 10,  # New: maximum batch size
) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Efficiently compute log P(paraphrase | context) in batches.

    Parameters
    ----------
    prompts      : List[str]
    paraphrase   : str
    model        : Hugging Face causal LM
    tokenizer    : Corresponding tokenizer
    max_tokens   : Max paraphrase tokens
    temperature  : Temperature scaling
    batch_size   : Maximum batch size for GPU memory management

    Returns
    -------
    logp_list       : List[float]
        Total log-probabilities of paraphrase for each prompt.
    token_lls_list  : List[torch.Tensor]
        Individual token-level log-likelihoods per paraphrase token.
    """
    device = next(model.parameters()).device
    dtype = torch.float16

    # Prepare concatenated inputs and tokenization
    concatenated_inputs = [p + paraphrase for p in prompts]
    
    # Tokenize concatenated sequences
    enc_concat = tokenizer(
        concatenated_inputs,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt"
    )
    
    # Tokenize prompts alone
    enc_prompts = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt"
    )

    # Setup DataLoader to handle batching
    dataset = TensorDataset(
        enc_concat.input_ids, enc_concat.attention_mask,
        enc_prompts.attention_mask
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    logp_list = []
    token_lls_list = []

    # Iterate over batches
    for batch in tqdm(dataloader, desc="Computing batched log-probs"):
        input_ids_batch, attn_mask_batch, prompt_mask_batch = [tensor.to(device) for tensor in batch]

        N, L_concat = input_ids_batch.size()

        with torch.autocast(device.type, enabled=True, dtype=dtype):
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attn_mask_batch,
                return_dict=True
            )

        logits = outputs.logits  # shape: (N, L_concat, vocab_size)

        for i in range(N):
            # Get actual prompt length from attention mask
            prompt_len = prompt_mask_batch[i].sum().item()
            paraphrase_start = prompt_len
            paraphrase_end = min(paraphrase_start + max_tokens, L_concat - 1)

            # Extract target paraphrase tokens
            target_ids = input_ids_batch[i, paraphrase_start:paraphrase_end + 1]

            # Get logits for paraphrase tokens (predicting next tokens)
            logits_for_paraphrase = logits[i, paraphrase_start - 1:paraphrase_end]

            # Compute log-probs
            log_probs = F.log_softmax(logits_for_paraphrase / temperature, dim=-1)
            paraphrase_log_probs = log_probs[torch.arange(target_ids.size(0)), target_ids]

            # Store results
            total_logp = paraphrase_log_probs.sum().item()
            logp_list.append(total_logp)
            token_lls_list.append(paraphrase_log_probs.cpu())

        # Cleanup per-batch
        del logits, outputs, input_ids_batch, attn_mask_batch, prompt_mask_batch
        torch.cuda.empty_cache()

    return logp_list, token_lls_list


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


def build_variants_for_entity_type_new(
    passage: str,
    entity_offsets: Dict[str, List[Tuple[int, int]]],
    ent_to_do: str,
    candidates: List[str],
    number_to_gen: int = 8,
    seed: int = 42,
    debug: bool = False,
):
    """
    Return `number_to_gen` passages where **only `ent_to_do`** is perturbed.

    • variants[0]  == original passage
    • variants[i]  (i>0) has every *unique* surface form of that entity
      replaced by a *different* random string (fresh draw per variant).

    The function never changes the length of non-target entities, so all
    offsets in `entity_offsets` remain valid.
    """
    if ent_to_do not in entity_offsets:
        raise ValueError(f"{ent_to_do=} not in entity_offsets")

    # ----- gather mentions for the target entity --------------------------
    mentions = sorted(entity_offsets[ent_to_do], key=lambda t: t[0])   # [(s,e),…]
    if not mentions:
        raise ValueError(f"No offsets for {ent_to_do}")

    # distinct surface forms and sanity-check candidate pool
    surf_forms = {passage[s:e] for s, e in mentions}
    u = len(surf_forms)
    if len(candidates) < u:
        raise ValueError(f"Need ≥{u} candidates, got {len(candidates)}")

    rng      = random.Random(seed)
    variants = [passage]                                              # #0

    for vidx in range(1, number_to_gen):
        # 1️⃣ sample a one-to-one mapping
        repls   = rng.sample(candidates, k=u)
        mapping = dict(zip(sorted(surf_forms), repls))

        if debug:
            print(f"\nVariant #{vidx} for {ent_to_do}:")
            for sf in sorted(surf_forms):
                print(f"  {sf!r}  →  {mapping[sf]!r}")

        # 2️⃣ rebuild the passage
        pieces   = []
        cursor   = 0              # start of yet-to-be-copied slice
        for s, e in mentions:
            pieces.append(passage[cursor:s])          # text BEFORE entity
            pieces.append(mapping[passage[s:e]])      # replacement
            cursor = e                                # advance cursor
        pieces.append(passage[cursor:])               # tail after last entity

        variants.append("".join(pieces))

    return variants

def attack_scores(
    logp_list: List[float],
    token_lls_list: List[torch.Tensor],
    k_percents: List[int] = (5, 10, 20, 30, 40),
    true_index: int = 0
) -> Dict[str, Dict]:
    """
    Parameters
    ----------
    logp_list        : list[float]
        log P(paraphrase | prompt_i) for every candidate i
    token_lls_list   : list[Tensor] | None
        token-level **log-probs** (1D tensor, len = #tokens) for every prompt
        → only needed for min-k% attacks.  Pass `None` if you only want
        <full-prob> and <ppl>.
    k_percents       : Which *k %* to evaluate for min-k attacks
    true_index       : position of the real (ground-truth) prompt (default 0)

    Returns
    -------
    dict
        {
          "full_prob": {"scores": [...], "winner": j, "correct": 0/1},
          "ppl":       {...},
          "min_k_05":  {...},
          "min_k_10":  {...},
          … etc …
        }
    """

    N = len(logp_list)
    out: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ PPL
    ppl_scores = [-lp / math.log(10)   # convert nats ➜ bits (or skip if not needed)
                  for lp in logp_list]                 # lower PPL ⇒ better
    win_idx = int(np.argmin(ppl_scores))
    out["ppl"] = {
        "scores":  ppl_scores,
        "winner":  win_idx,
        "correct": int(win_idx == true_index)
    }

    # ------------------------------------------------------------------ min-k %
    if token_lls_list is not None:
        L = len(token_lls_list[0])     # #tokens in paraphrase
        for k in k_percents:
            k_cnt   = max(1, int(round(k / 100 * L)))
            scores  = []
            for tok_ll in token_lls_list:
                # pick k tokens with *lowest* log-prob (most surprising)
                vals, _ = torch.topk(tok_ll, k_cnt, largest=False)
                scores.append(vals.mean().item())    # higher ⇒ less surprising
            win_idx = int(np.argmax(scores))
            key = f"min_k_{k:02d}"
            out[key] = {
                "scores":  scores,
                "winner":  win_idx,
                "correct": int(win_idx == true_index)
            }

    return out


# ===================================
# TOKEN REPLACEMENT AND PROCESSING
# ===================================

def process_entry(candidates, entry, output_entry, model, tokenizer, gpu_id=0,device_map=None):
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

    meta_save = {}

    passage_lines = entry["passage"]
    private_entities_all = entry["private_entities"]
    
    # Combine all lines into a single passage with offsets adjustment

    MAX_CHARS = 10000
    # MAX_CHARS = 5000

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

            tup = (s + going_offset, e + going_offset)
            if e_type not in offsets_new:
                offsets_new[e_type] = []
                offsets_new[e_type].append(tup)
            else:
                # Check if this offset already exists
                if tup not in offsets_new[e_type]:
                    offsets_new[e_type].append(tup)

            #offsets_new.setdefault(e_type, []).append((s + going_offset, e + going_offset))

        # Append the line
        passage_new += line_text + " "
        going_offset += line_len

    all_cand_variants = {}
    mappings = {}

    for ent in ENTITIES_TO_ATTACK:
        print(f"DOING FOR Entity type: {ent}")
        # We should create a new version of the passage_new that redacts other entities

        base_passage = passage_new

        # 2) we’ll build a redacted version for this ent
        redacted = base_passage
        
        print(f"The redacted passage for entity {ent} is:")
        print(redacted[:1000])
        print("====================================")

        # checking if this shit is okay;
        
        passage_variants = build_variants_for_entity_type_new(
            redacted,
            offsets_new,
            ent,
            candidates[ent],
            number_to_gen=CANDS,  # or any number you need
            seed = SEED[ent],
            debug=DEBUG_MODE)

        # if DEBUG_MODE:
        for line in passage_variants:
            print(f"   variant -->{line[:1000]}")
        
        all_cand_variants[ent] = passage_variants
        # mappings[ent] = mapping_ranges

    MAX_TOKENS = min(tokenizer(passage_new, return_tensors="pt")["input_ids"].shape[1],100)

    # Create prompt for the model
    prompt = format_prompt_new_template(tokenizer, passage_new, placeholder=
                                        PLACEHOLDER_TOKEN)

    # create prompt variants for all candidates

    all_can_prompts = {}
    for ent in all_cand_variants:
        all_can_prompts[ent] = []
        for passage in all_cand_variants[ent]:

            all_can_prompts[ent].append(format_prompt_new_template(tokenizer, passage, placeholder=PLACEHOLDER_TOKEN))


    #paraphrase = output_entry["redacted_text_split"]
    paraphrase = output_entry["output"]

    ent_results = {}
    ent_results_meta = {}

    for ent in all_can_prompts:
        ent_results[ent] = {}
        ent_results_meta[ent] = {}

        # Add the original passage as the last entry
        logp_list, token_lls_list = batched_logp_causal_single_pass_batched(
            all_can_prompts[ent],
            paraphrase,
            model,
            tokenizer,
            max_tokens = 900,
            temperature=1.0,
            batch_size=5            # ← same T you use elsewhere
        )

        # Get all probs
        logp = torch.tensor(logp_list)                           # shape (N,)
        logp_norm = logp - torch.logsumexp(logp, dim=0)          # normalized log-posterior
        post = logp_norm.exp().tolist()                          # now sums to 1
        # prnint this out properly
        if DEBUG_MODE:
            print(f"Logp: {logp}")
            print(f"Logp norm: {logp_norm}")
            print(f"Post: {post}")
        
        attack_scores_dict = attack_scores(
            logp_list,
            token_lls_list)

        if DEBUG_MODE:
            print(f"Attack scores for entity {ent}:")
            for key, value in attack_scores_dict.items():
                print(f"  {key}: {value}")

        # instead of saving the raw tensor...
        ent_results_meta[ent]["token_lls_list"] = [t.cpu().tolist() for t in token_lls_list]
        #ent_results_meta[ent]["candidate prompts"] = all_can_prompts[ent]
        # ent_results_meta[ent]["mappings"] = mappings[ent]

        ent_results[ent]["candidate prompts"] = all_can_prompts[ent]
        ent_results[ent]["attack_scores"] = attack_scores_dict
        ent_results[ent]["logp"]           = [float(v) for v in logp_list]
        ent_results[ent]["logp_norm"]      = [float(v) for v in logp_norm]
        ent_results[ent]["post"]           = [float(v) for v in post]

    return(ent_results, ent_results_meta)


def main():
    start_time = time.time()
    log(f"Starting processing for data indices {args.start} to {args.end}")

    # 1) load the candidate set once
    with open("candidate_set_100.json", "r") as jf:
        cand = json.load(jf)

    # 2) load the slice of the big input
    with open(INPUT_FILE, "r", encoding="utf-8") as jf:
        big_data = json.load(jf)
    data = big_data[args.start:args.end]
    log(f"Loaded {len(data)} entries from {INPUT_FILE}")

    # 3) load any existing output to resume
    if os.path.isfile(args.output_file) and os.path.getsize(args.output_file) > 0:
        with open(args.output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        log(f"Resuming from {len(output_data)} existing results")
    else:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, use_auth_token=HF_TOKEN, torch_dtype=torch.float16,
        device_map="auto" if args.gpu else None).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=HF_TOKEN)

    # 4) iterate *all* entries, passing in cand, input & old output
    results = []
    meta_results = []
    for i in tqdm(range(args.start,args.end)):
        entry = data[i]
        old_out = output_data[i]
        ent_results, ent_results_meta = process_entry(cand, entry, old_out,
                                     model, tokenizer)
        results.append(ent_results)
        meta_results.append(ent_results_meta)

        with open(ATTACK_OUTPUT_FILE_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(ATTACK_OUTPUT_FILE_PICKLE, "wb") as f:
            pickle.dump(results, f)
        with open(ATTACK_OUTPUT_FILE_META_DATA, "wb") as f:
            pickle.dump(meta_results, f)
        
if __name__ == "__main__":
    main()