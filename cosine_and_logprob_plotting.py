# pip install torch transformers matplotlib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from typing import List, Dict, Any, Sequence
import math
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def _to_list_1d(x: Sequence[float]) -> List[float]:
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.ravel().tolist()
    except Exception:
        pass
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.reshape(-1).detach().cpu().tolist()
    except Exception:
        pass
    return list(map(float, x))

def _prefix(tokens: Sequence[str], start_idx: int, k: int = 5) -> List[str]:
    s = max(0, start_idx - k)
    return list(tokens[s:start_idx])

def _follow(tokens: Sequence[str], idx: int, k: int = 5) -> List[str]:
    # returns next k tokens strictly after idx
    return list(tokens[idx + 1: idx + 1 + k])

def extract_high_sim_blocks(
    tokens: Sequence[str],
    cos_sims: Sequence[float],
    threshold: float,
    min_tokens: int = 2,
    drop_nan: bool = True,
    inclusive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Finds contiguous token spans where every edge similarity meets the threshold.

    Args
    ----
    tokens: length T
    cos_sims: length T-1; cos_sims[i] = sim between tokens[i] and tokens[i+1]
    threshold: similarity gate
    min_tokens: keep spans with at least this many tokens (>=2)
    drop_nan: NaNs break runs if True; otherwise NaNs must pass comparator (they won't)
    inclusive: if True, uses >= threshold; else uses > threshold

    Returns
    -------
    List of dicts with:
      - start_idx, end_idx (inclusive token indices)
      - tokens (slice)
      - mean_cos (mean of edges inside the block)
      - edge_span [e_start, e_end]
      - length (token count)
    """
    toks = list(tokens)
    sims = _to_list_1d(cos_sims)

    T = len(toks)
    assert len(sims) == T - 1, f"cos_sims must have length T-1 (got {len(sims)} vs {T-1})"
    assert min_tokens >= 2, "min_tokens must be >= 2"

    def passes(x: float) -> bool:
        if math.isnan(x):
            return False if drop_nan else False  # NaN never passes comparator
        return (x >= threshold) if inclusive else (x > threshold)

    blocks = []
    i = 0  # edge index
    while i < len(sims):
        s = None
        # start of a high-sim run
        while i < len(sims):
            if passes(sims[i]):
                s = i
                break
            i += 1
        if s is None:
            break

        # extend run
        j = s
        while j + 1 < len(sims) and passes(sims[j + 1]):
            j += 1

        # edge run s..j => token span [s, j+1]
        start_tok, end_tok = s, j + 1
        length = end_tok - start_tok + 1
        if length >= min_tokens:
            edges = sims[s:j+1]
            finite = [x for x in edges if not math.isnan(x)]
            mean_cos = sum(finite) / len(finite) if finite else float("nan")
            blocks.append({
                "start_idx": start_tok,
                "end_idx": end_tok,
                "tokens": toks[start_tok:end_tok+1],
                "mean_cos": mean_cos,
                "edge_span": [s, j],
                "length": length,
            })

        i = j + 1

    return blocks

def extract_low_logprob_tokens(
    tokens: Sequence[str],
    token_logprobs_shifted: Sequence[float],
    threshold: float,
    prefix_k: int = 5,
    follow_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Flags predicted tokens with log-prob < threshold.
    token_logprobs_shifted aligns with tokens[1:]; index i in [1..T-1] is the predicted token.
    Adds 'prefix_tokens' and 'following_tokens' around each flagged token.
    """
    toks = list(tokens)
    lps = _to_list_1d(token_logprobs_shifted)
    T = len(toks)
    assert len(lps) == T - 1, f"logprobs must have length T-1 (got {len(lps)} vs {T-1})"

    results = []
    for i in range(1, T):  # token index of the predicted token
        lp = lps[i - 1]
        if not math.isnan(lp) and lp < threshold:
            results.append({
                "idx": i,
                "token": toks[i],
                "logprob": lp,
                "prefix_tokens": _prefix(toks, i, prefix_k),
                "following_tokens": _follow(toks, i, follow_k),
            })
    return results

def curvature_from_states(states, eps=1e-12, degrees=True):
    X = states if isinstance(states, torch.Tensor) else torch.tensor(states)
    V = X[1:] - X[:-1]                      # [T-1, D]
    v_k, v_k1 = V[:-1], V[1:]               # [T-2, D], [T-2, D]
    num = (v_k1 * v_k).sum(-1)
    n1 = v_k1.norm(dim=-1).clamp_min(eps)
    n0 = v_k.norm(dim=-1).clamp_min(eps)
    cos_t = (num / (n1 * n0)).clamp(-1+1e-7, 1-1e-7)
    ang = torch.acos(cos_t)
    return ang * (180.0 / math.pi) if degrees else ang

if __name__ == "__main__":
    # record_path = "/scratch/rst306/verifier_scaling/verifiers/result/aime/Qwen_Qwen3-0.6B/20251002_134510/record.json"
    qwen_8b_path = "/scratch/rst306/verifier_scaling/verifiers/result/aime/Qwen_Qwen3-8B/20251010_115527/record.json"
    record_path = qwen_8b_path    
    with open(record_path, 'r') as fp:
        records = json.load(fp)
    correct = [r for r in records if r['solver_correct']]
    all_correct_outputs = [r['solver_full_output'] for r in correct]
    # idx = 1
    MODEL_NAME = "Qwen/Qwen3-8B" #"Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
    model.eval()

    for idx, TEXT in tqdm(enumerate(all_correct_outputs)):
        # --- config ---
        # TEXT = "In the beginning the universe was created. This has made a lot of people very angry and been widely regarded as a bad move."
        TEXT = all_correct_outputs[idx]
        # --- load ---

        # --- tokenize ---
        enc = tokenizer(TEXT, return_tensors="pt").to("cuda")
        input_ids = enc["input_ids"]          # [1, T]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # Final hidden states per position (the "output token embeddings")
            # For AutoModelForCausalLM, `out` exposes:
            #   - logits: [1, T, V]
            #   - (optionally) hidden_states if enabled; but last_hidden_state is always available on `model.base_model(...)`.
            # To get last_hidden_state reliably, call the base model:
            base_out = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            H = base_out.last_hidden_state    # [1, T, D]
            logits = out.logits               # [1, T, V]

        # --- cosine similarities between adjacent output token embeddings ---
        # Compare embedding at i+1 with embedding at i
        # Result shape: [T-1]
        cos_sims = F.cosine_similarity(H[:, 1:, :], H[:, :-1, :], dim=-1).squeeze(0)  # [T-1]

        # --- token log-probabilities (proper causal shift) ---
        # Logits at position i predict token at position i+1
        # We ignore the final position since it predicts "the next" after the last input token.
        logprobs_all = F.log_softmax(logits[:, :-1, :], dim=-1)         # [1, T-1, V]
        target_ids = input_ids[:, 1:]                                   # [1, T-1]
        token_logprobs = logprobs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # [T-1]

        # --- pretty x-axis labels (token strings aligned to predicted tokens) ---
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        x_positions = list(range(1, len(tokens)))   # positions of predicted tokens (1..T-1)
        x_labels = tokens[1:]                        # tokens that were predicted

        def escape_dollars_fig(fig):
            for ax in fig.axes:
                # texts + ticklabels + title/axis labels
                texts = list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
                texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
                for t in texts:
                    s = t.get_text()
                    if '$' in s:
                        t.set_text(s.replace('$', r'\$'))


        save_dir = f"plots/{MODEL_NAME}/AIME_correct_{idx}"
        os.makedirs(save_dir, exist_ok=True)
        # --- plots ---
        # 1) Cosine similarity
        fig = plt.figure(figsize=(40,8))
        plt.plot(x_positions, cos_sims.cpu().tolist(), marker="o")
        plt.title("Adjacent Output-Embedding Cosine Similarities")
        plt.xlabel("Token index (token predicted at position i)")
        plt.ylabel("cosine_similarity(H[i], H[i-1])")
        xlabels = [lab.replace("$", "\$") for lab in x_labels]
        # plt.xticks(x_positions, x_labels, rotation=90)

        plt.tight_layout()
        # plt.show()
        fig.savefig(f"{save_dir}/cosine_similarities.png")

        plt.close(fig)
        plt.cla()
        plt.clf()
        # 2) Token log-probabilities
        fig = plt.figure(figsize=(30,8))
        plt.plot(x_positions, token_logprobs.cpu().tolist(), marker="o")
        plt.title("Per-Token Log-Probabilities (shifted)")
        plt.xlabel("Token index (token predicted at position i)")
        plt.ylabel("log p(token_i | <prefix>)")
        xlabels = [lab.replace("$", "\$") for lab in x_labels]

        # plt.xticks(x_positions, x_labels, rotation=90)

        plt.tight_layout()
        # plt.show()
        fig.savefig(f"{save_dir}/logprobs.png")
        plt.close(fig)
        plt.cla()
        plt.clf()
        # --- optional: package results programmatically ---
        results = {
            "tokens": x_labels,
            "cosine_sim": cos_sims.cpu().tolist(),
            "logprob": token_logprobs.cpu().tolist(),
        }
        # print(results)
        blocks = extract_high_sim_blocks(tokens, cos_sims, threshold=0.8)
        lengths_to_blocks = defaultdict(list)

        low_prob_tokens = extract_low_logprob_tokens(tokens, token_logprobs, threshold=-5.0)
        print("low prob")
        print(low_prob_tokens)
        for block in blocks:
            # print(block["tokens"], block["length"])
            lengths_to_blocks[block["length"]].append(block["tokens"])

        with open(f"{save_dir}/blocks.json", 'w') as fp:
            json.dump(lengths_to_blocks, fp)

        with open(f"{save_dir}/low_prob_tokens.json", 'w') as fp:
            json.dump(low_prob_tokens, fp)

        curv = curvature_from_states(H.squeeze(0), degrees=True)  # [T-2]
        x_positions = list(range(1, len(tokens)-1))
        x_labels = tokens[1:-1]

        fig = plt.figure(figsize=(50,10))
        plt.plot(x_positions, curv.cpu().tolist(), marker="o")
        plt.title("Curvature across output embeddings")
        plt.xlabel("Token index (center of angle)")
        plt.ylabel("degrees")
        # plt.xticks(x_positions, x_labels, rotation=90)
        
        plt.tight_layout()
        # plt.show()
        fig.savefig(f"{save_dir}/curvature.png")
        # break