from typing import Sequence, List, Union, Literal, Tuple, Dict, Any, Iterable, Optional
from collections import defaultdict
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
import os
from tqdm import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# Try to import hdbscan; fallback to DBSCAN
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    from sklearn.cluster import DBSCAN
    HAVE_HDBSCAN = False


def cluster_states(X: np.ndarray, n_states: int = 6, random_state: int = 0):
    km = KMeans(n_clusters=n_states, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km

def cluster_states_hdbscan(X: np.ndarray,
                           min_cluster_size: int = 30,
                           min_samples: int | None = None,
                           random_state: int = 0):
    """
    Standardize -> PCA to pca_dims -> HDBSCAN.
    Returns labels (N,), reducer (PCA), and the clusterer.
    Noise points get label -1.
    """
    Xp = PCA(n_components=3, random_state=random_state).fit_transform(normalize(X, axis=1))
    if HAVE_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(Xp)
    else:
        # Fallback: DBSCAN with a simple eps heuristic (use kNN distance)
        from sklearn.neighbors import NearestNeighbors
        k = max(5, int(np.sqrt(Xp.shape[1])))
        nn = NearestNeighbors(n_neighbors=k).fit(Xp)
        dists, _ = nn.kneighbors(Xp)
        kth = np.median(dists[:, -1])
        eps = 1.5 * kth
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels = clusterer.fit_predict(Xp)
        # map labels to consecutive ints with noise=-1 preserved
        uniq = sorted([l for l in np.unique(labels) if l != -1])
        remap = {l:i for i,l in enumerate(uniq)}
        labels = np.array([remap.get(l, -1) for l in labels], dtype=int)
    return labels, Xp, clusterer

def relabel_noise_to_nearest(X: np.ndarray, labels: np.ndarray):
    """
    Assign -1 (noise) to nearest cluster centroid in original space.
    If all points are noise, return labels unchanged.
    """
    noise_idx = np.where(labels == -1)[0]
    if noise_idx.size == 0:
        return labels
    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    if len(cluster_ids) == 0:
        return labels
    centroids = np.stack([X[labels == c].mean(axis=0) for c in cluster_ids], axis=0)
    # assign each noise point to nearest centroid
    diff = X[noise_idx][:, None, :] - centroids[None, :, :]
    d2 = np.sum(diff*diff, axis=2)
    assign = np.argmin(d2, axis=1)
    new_labels = labels.copy()
    new_labels[noise_idx] = np.array(cluster_ids)[assign]
    return new_labels

def build_msm(labels: np.ndarray, n_states: int, lag: int = 1):
    C = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(labels) - lag):
        i, j = labels[t], labels[t + lag]
        C[i, j] += 1
    T = C.copy()
    row_sums = T.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() != 0
    T[mask] = T[mask] / row_sums[mask]
    zero_idx = np.where(~mask)[0]
    for i in zero_idx:
        T[i, i] = 1.0
    # stationary distribution
    pi = np.ones(n_states) / n_states
    for _ in range(500):
        pi_next = pi @ T
        if np.allclose(pi_next, pi, atol=1e-12):
            break
        pi = pi_next
    pi /= pi.sum()
    return T, C, pi

def plot_T_heatmap(T, title="MSM transition matrix"):
    fig = plt.figure(figsize=(6,5), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(T, origin='lower', aspect='auto')
    ax.set_xlabel("to state j"); ax.set_ylabel("from state i"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); return fig

def plot_state_graph(T, pi, min_edge=0.03, title="MSM state graph"):
    try:
        import networkx as nx
    except Exception:
        return None
    n = T.shape[0]; G = nx.DiGraph()
    for i in range(n): G.add_node(i, weight=pi[i])
    for i in range(n):
        for j in range(n):
            if T[i,j] >= min_edge: G.add_edge(i, j, weight=T[i,j])
    try:
        pos = nx.kamada_kawai_layout(G, weight='weight')
    except Exception:
        pos = nx.circular_layout(G)
    node_sizes = 3000 * (pi / (pi.max() if pi.max()>0 else 1.0))
    edge_widths = [4.0 * d['weight'] for _,_,d in G.edges(data=True)]
    fig = plt.figure(figsize=(7,6), dpi=140); ax = fig.add_subplot(111)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"{i}\nπ={pi[i]:.2f}"} , font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, arrowstyle='-|>', ax=ax)
    elabs = {(i,j): f"{T[i,j]:.2f}" for i,j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabs, font_size=7, ax=ax)
    ax.set_title(title); ax.axis('off'); plt.tight_layout(); return fig

def finite_difference_velocity(X: np.ndarray, dt: float = 1.0, scheme: str = "central") -> np.ndarray:
    """Velocity for trajectory X∈R^{N×D}. Returns v with shape (N, D)."""
    N, D = X.shape
    v = np.zeros_like(X, dtype=float)
    if scheme == "forward":
        v[:-1] = (X[1:] - X[:-1]) / dt
        v[-1] = v[-2]
    elif scheme == "backward":
        v[1:] = (X[1:] - X[:-1]) / dt
        v[0] = v[1]
    else:  # central
        if N >= 3:
            v[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
            v[0] = (X[1] - X[0]) / dt
            v[-1] = (X[-1] - X[-2]) / dt
        elif N == 2:
            v[0] = (X[1] - X[0]) / dt
            v[1] = v[0]
        else:
            raise ValueError("Need at least 2 points to compute velocity.")
    return v

def project_positions_and_velocity(X: np.ndarray, v: np.ndarray, n_components: int = 3):
    """Fit PCA on X, project X→X3 and v→v3 with same linear map."""
    pca = PCA(n_components=n_components)
    X3 = pca.fit_transform(normalize(X,axis=1))          # (N, 3)
    v3 = v @ pca.components_.T         # derivative ignores mean
    return X3, v3, pca

def plot_flow_3d(X: np.ndarray, dt: float = 1.0, step_arrows: int = 1, title: str = "3D Flow Map"):
    """
    X: (N, D). Computes velocity, projects to 3D via PCA, plots polyline + velocity quivers.
    """
    assert step_arrows >= 1
    N = X.shape[0]
    v = finite_difference_velocity(X, dt=dt, scheme="central")
    X3, v3, _ = project_positions_and_velocity(X, v, n_components=3)

    # subsample arrows
    if N <= 200:
        idx = np.arange(0, N, step_arrows)
    else:
        stride = max(step_arrows, N // 100)
        idx = np.arange(0, N, stride)

    # normalize arrow directions, scale by robust speed
    vnorm = np.linalg.norm(v3[idx], axis=1, keepdims=True)
    vhat = np.divide(v3[idx], np.where(vnorm == 0, 1, vnorm))
    speeds = np.linalg.norm(v3, axis=1)
    robust_scale = np.percentile(speeds, 90) if speeds.size else 1.0
    length = 0.25 * robust_scale if robust_scale > 0 else 0.1

    t = np.linspace(0, 1, N)

    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X3[:, 0], X3[:, 1], X3[:, 2], linewidth=1.0, alpha=0.8)
    sc = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=t, s=10)
    ax.quiver(X3[idx, 0], X3[idx, 1], X3[idx, 2],
              vhat[:, 0], vhat[:, 1], vhat[:, 2],
              length=length, normalize=False)

    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Normalized time")
    plt.tight_layout()
    return fig, ax

def pca_trajectory_3d(X: np.ndarray):
    pca = PCA(n_components=3, svd_solver="auto", random_state=0)

    Z = pca.fit_transform(normalize(X, axis=1))
    return Z, pca

def plot_trajectory_3d(Z: np.ndarray, start_marker_size=30, end_marker_size=30):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Z[:,0], Z[:,1], Z[:,2])
    ax.scatter(Z[0,0], Z[0,1], Z[0,2], s=start_marker_size)
    ax.scatter(Z[-1,0], Z[-1,1], Z[-1,2], s=end_marker_size)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA trajectory")
    return fig, ax


def get_token_span(text: str, substring: str, tokenizer: AutoTokenizer) -> Tuple[int,int,List[int],List[str],List[Tuple[int,int]]]:
    """
    Returns (token_start, token_end_inclusive, token_ids[token_start:token_end+1],
             tokens[token_start:token_end+1], offsets[token_start:token_end+1])
    Raises ValueError if substring not found or tokenizer is not a fast tokenizer.
    """
    if not tokenizer.is_fast:
        raise ValueError("Tokenizer must be a fast tokenizer (use_fast=True).")
    char_start = text.find(substring)
    if char_start == -1:
        raise ValueError("Substring not found in text.")
    char_end = char_start + len(substring)  # exclusive

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets: List[Tuple[int,int]] = enc['offset_mapping']  # list of (start,end) per token
    input_ids: List[int] = enc['input_ids']
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    # Find tokens that overlap the substring character span
    token_indices = [i for i,(s,e) in enumerate(offsets) if not (e <= char_start or s >= char_end)]
    if not token_indices:
        # Rare: possibly substring aligns with a special token (offset (0,0)).
        # Try matching on tokens' decoded text as fallback.
        joined = tokenizer.decode(input_ids)
        raise ValueError("No token offsets overlap substring. Tokenizer offsets: fallback required.")

    token_start, token_end = min(token_indices), max(token_indices)
    return token_start, token_end, input_ids[token_start:token_end+1], tokens[token_start:token_end+1], offsets[token_start:token_end+1]


def extract_hidden_representations(
    model,
    tokenizer,
    text: str,
    substring: str,
    device: str = "cpu",
    keep_layer_outputs: bool = True
) -> Dict[str, np.ndarray]:
    """
    Loads tokenizer+model, finds token span for substring, runs model, returns hidden reps.
    Returns dict:
      {
        "token_start": int,
        "token_end": int,
        "tokens": [str],
        "input_ids": [int],
        "layer_{i}": np.ndarray(shape=(num_tokens_in_span, hidden_size))  # i=0..L (0 = embeddings)
      }
    Notes:
    - Uses AutoTokenizer (fast) and AutoModel (base model) from transformers.
    - Caller can change model_name (e.g. "gpt2", "bert-base-uncased", "facebook/opt-125m").
    """

    # Tokenize once with offsets & return PyTorch tensors for model input
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    enc = {k: v.to(device) for k,v in enc.items() if k != "offset_mapping"}
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)['offset_mapping']

    # Find token span
    token_start, token_end, token_ids_slice, tokens_slice, slice_offsets = get_token_span(text, substring, tokenizer)

    # Forward pass: request hidden states
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple: (embedding_output, layer1, layer2, ...)

    # Collect representations: for each layer, extract tokens in [token_start:token_end+1]
    result = {
        "token_start": token_start,
        "token_end": token_end,
        "tokens": tokens_slice,
        "input_ids": token_ids_slice,
        "offsets": slice_offsets
    }

    for layer_idx, layer_tensor in enumerate(hidden_states):
        # layer_tensor shape: (batch=1, seq_len, hidden_dim)
        arr = layer_tensor[0, token_start:token_end+1, :].cpu().numpy().copy()  # shape (num_tokens, hidden_dim)
        result[f"layer_{layer_idx}"] = arr

    return result

def recurrence_plot(Z: np.ndarray):
    """Distance recurrence image (reveals returns, cycles)."""
    from scipy.spatial.distance import cdist
    D = cdist(Z, Z, metric="euclidean")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(D, origin="lower", aspect="auto")
    ax.set_title("Recurrence (distance) plot")
    plt.colorbar(im, ax=ax)
    # plt.show()
    return fig, ax


with open("/scratch/rst306/action_abstractions/action_abstraction/abstraction_results/Qwen/Qwen3-1.7B/abstractions.json", 'r') as fp:
    abstraction_outputs = json.load(fp)

MODEL_NAME = "Qwen/Qwen3-8B" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to("cuda")
model.eval()

for i, output in tqdm(enumerate(abstraction_outputs)):
    
    matched_abstractions = [abstract for abstract in output["abstractions"] if len(abstract["matched_subsequences"]) > 0]
    #"Qwen/Qwen3-0.6B"  
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
    # model.eval()
    orig_text = output["solver_output"]
    for j, matched in enumerate(matched_abstractions):
        
        # print(hiddens)
        # print(hiddens["layer_1"], hiddens["layer_1"].shape)
        save_dir = f"trajectory_results/{MODEL_NAME}/correct_{i}/abstraction_{j}"
        recurrence_save_dir = f"recurrence_plot_results/{MODEL_NAME}/correct_{i}/abstraction_{j}"
        states_save_dir = f"state_plot_results/{MODEL_NAME}/correct_{i}/abstraction_{j}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(recurrence_save_dir, exist_ok=True)
        os.makedirs(states_save_dir, exist_ok=True)

        for save_directory in [save_dir, recurrence_save_dir, states_save_dir]:
            with open(f"{save_directory}/abstraction.json", 'w') as fp:
                json.dump(matched, fp)

        for k, m_subseq in enumerate(matched["matched_subsequences"]):
            hiddens = extract_hidden_representations(model, tokenizer, orig_text, m_subseq["match"], device="cuda")
            save_dir = f"{save_dir}/match_{k}"
            recurrence_save_dir = f"{recurrence_save_dir}/match_{k}"
            states_save_dir = f"{states_save_dir}/match_{k}"
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(recurrence_save_dir, exist_ok=True)
            os.makedirs(states_save_dir, exist_ok=True)
            for save_directory in [save_dir, recurrence_save_dir, states_save_dir]:
                with open(f"{save_directory}/match.json", 'w') as fp:
                    json.dump(m_subseq, fp)

            for key in hiddens:
                if "layer" in key:
                    layer_array = hiddens[key]
                    Z, pca = pca_trajectory_3d(layer_array)
                    print(pca.explained_variance_ratio_)
                    fig, ax = plot_trajectory_3d(Z)
                    fig.savefig(f"{save_dir}/trajectory_{key}.png")
                    plt.close(fig)

                    fig, ax = recurrence_plot(Z)
                    fig.savefig(f"{recurrence_save_dir}/trajectory_{key}.png")
                    plt.close(fig)
                    labels_hdb, Xp, clusterer = cluster_states_hdbscan(layer_array, min_cluster_size=10, min_samples=None)
                    labels_filled = relabel_noise_to_nearest(layer_array, labels_hdb)
                    K = int(labels_filled.max()+1)

                    # MSM
                    try:
                        T, C, pi = build_msm(labels_filled, n_states=K, lag=1)
                        fig2 = plot_T_heatmap(T, title="MSM transition matrix (lag=3)")
                        fig3 = plot_state_graph(T, pi, min_edge=0.05, title="MSM state graph (edge ≥ 0.05)")
                        fig2.savefig(f"{states_save_dir}/msm_transition_trajectory_{key}.png")
                        fig3.savefig(f"{states_save_dir}/msm_state_graph_trajectory_{key}.png")
                        plt.close(fig2)
                        plt.close(fig3)

                    except IndexError:
                        print("Not enough states")


                # fig, ax = plot_flow_3d(layer_array)
                # fig.savefig(f"{flow_save_dir}/trajectory_{key}.png")
                # plt.close(fig)
        # Example usage:
        # if __name__ == "__main__":
        #     model_name = "gpt2"            # change to desired model
        #     text = "The quick brown fox jumps over the lazy dog."
        #     substring = "brown fox"

        #     out = extract_hidden_representations(model_name, text, substring, device="cpu")
        #     print("Token span:", out["token_start"], out["token_end"])
        #     print("Tokens:", out["tokens"])
        #     print("Layer 0 (embeddings) shape:", out["layer_0"].shape)
        #     print("Last layer shape:", out[f"layer_{len([k for k in out.keys() if k.startswith('layer_')])-1}"].shape)

        #     # Example: compute mean representation across span at final layer
        #     final_layer_idx = max(int(k.split("_")[1]) for k in out.keys() if k.startswith("layer_"))
        #     span_mean = out[f"layer_{final_layer_idx}"].mean(axis=0)  # (hidden_dim,)
        #     print("Mean final-layer vector (len):", span_mean.shape[0])
