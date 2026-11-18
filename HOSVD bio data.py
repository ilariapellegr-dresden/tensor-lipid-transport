# Higher Order Singular Value decomposition of a tensor
# Pipeline for: Tensor analysis for lipid transport 
# MPI-CBG, September 2025 


import os
import gc
from typing import Optional, Tuple, Dict

import numpy as np
from numpy import linalg
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
tl.set_backend('numpy')

# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler


# mapping for the names of the lipids 
LIPID_MAPPING = {
    "AL146": "SM(d18:1/Y16)",
    "KB75":  "PC(Y16/16:0)",
    "KB77":  "PC(Y16/18:1)",
    "KB116": "PA(18:1/Y16)",
    "AL136": "PC(16:0/Y16)",
    "AL140": "PC(18:1/Y16)",
    "KB79":  "PC(Y16/20:4)",
    "KB84":  "PC(20:4/Y16)"
}


def flattening(X, mode): 
    I, J, K = X.shape 
    if mode == 0: # lipid mode 
        return X.reshape(I, J*K) 
    elif mode == 1: # organelle mode 
        return np.transpose(X, (1, 0, 2)).reshape(J, I*K) 
    elif mode == 2: # time mode 
        return np.transpose(X, (2, 0, 1)).reshape(K, I*J) 
    else: 
        raise ValueError("Invalid mode: choose an integer between 0,1,2") 


# analyze hosvd results 

# cluster lipids that co-vary together, based on the embedding with the entries of the core as weights 
# optional input: "lipids" can be a list with the true names of the lipids 

def cluster_lipids(U0, core, number_name_file, lipids=None):
    importance = np.array([np.linalg.norm(core[p, :, :]) for p in range(core.shape[0])])  
    lipid_embeddings = U0 * importance[np.newaxis, :]  

    Z = linkage(lipid_embeddings, method="ward") # clustering with ward method

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=lipids if lipids is not None else np.arange(U0.shape[0]), leaf_rotation=90)
    plt.title("Hierarchical clustering of lipids ")
    plt.savefig("cluster_%d.svg" % (number_name_file), format="svg", bbox_inches="tight")
    plt.close()




# heatmap to represent the strength of the temporal evolution of the pairs (lipid, organelle)
# optional inputs : two lists with the names of the lipids and of the organelles
def temporal_strength_heatmap(U0, U1, U2, core, number_name_file, lipids=None, organelles=None):
    I, r0 = U0.shape
    J, r1 = U1.shape

    strengths = np.zeros((I, J))
    
    # for each (lipid, organelle) pair, reconstruct the temporal profile via U2
    for i in range(I):
        for j in range(J):
            coeffs = np.tensordot(np.tensordot(core, U0[i, :], axes=(0, 0)),
                                  U1[j, :], axes=(0, 0))  # shape (r2,)
            # linear combination of temporal components
            time_series = U2 @ coeffs  # shape (K,)
            # time_series = (time_series - np.mean(time_series)) / np.std(time_series) # standardization

            # compute temporal strength = norm of consecutive diffs 
            diffs = np.diff(time_series)
            strengths[i, j] = np.linalg.norm(diffs)
    
    # plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(strengths,
                xticklabels=organelles if organelles is not None else np.arange(J),
                yticklabels=lipids if lipids is not None else np.arange(I),
                cmap="viridis")
    plt.xlabel("Organelle")
    plt.ylabel("Lipid")
    plt.savefig("heatmap_%d.svg" % (number_name_file), format="svg", bbox_inches="tight")
    plt.close()




# perform a bootstrapping approach to incorporate the measurement error in the analysis




def _find_std_column(df: pd.DataFrame):
    candidates = ['Std', 'StdDev', 'SD', 'std', 'stddev', 'Std_error', 'SE', 'StdDev.']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'std' in c.lower() or (c.lower().endswith('sd') and len(c) > 2):
            return c
    return None

def unfolding(X: np.ndarray, mode: int) -> np.ndarray:
    I, J, K = X.shape
    if mode == 0:
        return X.reshape(I, J*K)
    elif mode == 1:
        return np.transpose(X, (1, 0, 2)).reshape(J, I*K)
    elif mode == 2:
        return np.transpose(X, (2, 0, 1)).reshape(K, I*J)
    else:
        raise ValueError("mode must be 0,1,2")


def hosvd_tensorly(
    X: np.ndarray,
    ranks: Optional[Tuple[int,int,int]] = None,
    n_iter_max: int = 200,
    tol: float = 1e-7
):
    core, factors = tucker(
        X,
        rank=ranks,
        n_iter_max=n_iter_max,
        tol=tol
    )
    U0, U1, U2 = factors
    return U0.astype(np.float32), U1.astype(np.float32), U2.astype(np.float32), core.astype(np.float32)


def compute_strengths_from_hosvd(U0, U1, U2, core):
    I, r0 = U0.shape
    J, r1 = U1.shape

    strengths = np.zeros((I, J), dtype=np.float32)

    for i in range(I):
        for j in range(J):
            coeffs = np.tensordot(np.tensordot(core, U0[i,:], axes=(0,0)), U1[j,:], axes=(0,0))
            time_series = U2 @ coeffs

            # ts_std = np.std(time_series)
            # if ts_std == 0:
                # strengths[i, j] = 0.0
                # continue

            # time_series = (time_series - np.mean(time_series)) / ts_std
            diffs = np.diff(time_series)
            strengths[i, j] = np.linalg.norm(diffs)

    return strengths


def bootstrap_hosvd(
    X: np.ndarray,
    std_X: np.ndarray,
    M_runs: int = 200,
    ranks: Optional[Tuple[int,int,int]] = None,
    what: str = "strengths",                 # "strengths" or "rates"
    seed: Optional[int] = None,
    verbose: bool = True,
    store_u0: bool = True                    # store per-run U0 embeddings
) -> Dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    I, J, K = X.shape

    mask = ~np.isnan(X)
    X_filled = np.nan_to_num(X, nan=0.0)

    if what == "strengths":
        per_shape = (M_runs, I, J)
    else:
        per_shape = (M_runs, I, J*J)

    # prepare the storage
    all_stats = np.zeros(per_shape, dtype=np.float32)
    if store_u0:
        # we don't know r0 until first run; so we'll collect in list then stack
        U0_list = []
    else:
        U0_list = None

    # bootstrap
    for m in range(M_runs):
        noise = np.zeros_like(X, dtype=np.float32)
        noise[mask] = rng.normal(loc=0.0, scale=std_X[mask])

        X_sample = X_filled + noise

        U0, U1, U2, core = hosvd_tensorly(X_sample, ranks=ranks)

        if what == "strengths":
            stats = compute_strengths_from_hosvd(U0, U1, U2, core).astype(np.float32)

        # store per-run stats
        all_stats[m, ...] = stats

        # store U0 if requested
        if store_u0:
            # append to list; we'll stack later
            U0_list.append(U0.astype(np.float32))

        if verbose and (m+1) % max(1, M_runs//10) == 0:
            print(f"Bootstrap run {m+1}/{M_runs} completed.")

    # aggregation
    mean = np.mean(all_stats, axis=0)
    # percentiles for empirical confidence intervals
    p12_5 = np.percentile(all_stats, 12.5, axis=0) # we are choosing conf level alpha = 0.25
    p87_5 = np.percentile(all_stats, 87.5, axis=0)
    std = np.std(all_stats, axis=0, ddof=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.inf)

    out = {
        'mean': mean,
        'p12_5': p12_5,
        'p87_5': p87_5,
        'std': std,
        'cv': cv,
    }

    # attach samples
    out['samples'] = all_stats
    # attach U0 storage
    if store_u0:
        out['U0_list'] = np.stack(U0_list, axis=0)  # (M_runs, I, r0)

    return out

# create the tensor
def build_tensor_and_std_from_df(
    df: pd.DataFrame,
    value_col: str = "Mean",
    std_col: Optional[str] = None,
    index_cols: Tuple[str,str,str] = ("Lipid", "Organelle", "Time"),
    sort_time: bool = True
):
    if std_col is None:
        std_col = _find_std_column(df)
    if std_col is None:
        raise ValueError("Cannot find a standard deviation column.")

    lipids = sorted(df[index_cols[0]].unique())
    organelles = sorted(df[index_cols[1]].unique())
    times = sorted(df[index_cols[2]].unique())

    I, J, K = len(lipids), len(organelles), len(times)

    lipid_to_idx = {l:i for i,l in enumerate(lipids)}
    org_to_idx = {o:j for j,o in enumerate(organelles)}
    time_to_idx = {t:k for k,t in enumerate(times)}

    X = np.full((I,J,K), np.nan, dtype=np.float32)
    STD = np.full((I,J,K), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        i = lipid_to_idx[row[index_cols[0]]]
        j = org_to_idx[row[index_cols[1]]]
        k = time_to_idx[row[index_cols[2]]]
        X[i,j,k] = row[value_col]
        STD[i,j,k] = row[std_col]

    # normalize the tensor with repsect to each lipid 
    for i in range(I):
        frob = np.linalg.norm(np.nan_to_num(X[i, :, :], nan=0.0), ord='fro')
        if frob > 0:
            X[i, :, :] /= frob
            STD[i, :, :] /= frob 

    med = np.nanmedian(STD)
    if np.isnan(med):
        med = 0.0
    STD = np.where(np.isnan(STD), med, STD)

    return X, STD, lipids, organelles, times

# results 

def plot_strengths_heatmap_CI(res, lipids, organelles, title_prefix="", out_dir="."):
    lb = res.get("p12_5")
    ub = res.get("p87_5")

    # huniform chromatic scale
    vmin = min(np.nanmin(lb), np.nanmin(ub))
    vmax = max(np.nanmax(lb), np.nanmax(ub))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(lb, xticklabels=organelles, yticklabels=lipids, cmap="coolwarm", ax=axes[0], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} 12.5 % (Lower Bound)")

    sns.heatmap(ub, xticklabels=organelles, yticklabels=lipids, cmap="coolwarm", ax=axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix} 87.5 % (Upper Bound)")

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{title_prefix.replace(' ', '_')}_CI.png")
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    gc.collect()
    return fname


def animate_bootstrap_strengths(res, lipids, organelles,
                                title="Bootstrap Animation",
                                out_fname="bootstrap_full.gif",
                                max_frames=60,
                                interval=200):

    samples = res.get('samples')

    M, I, J = samples.shape
    if M <= max_frames:
        frame_idx = np.arange(M)
    else:
        frame_idx = np.linspace(0, M-1, max_frames, dtype=int)

    vmin = float(np.min(samples))
    vmax = float(np.max(samples))

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(fr):
        ax.clear()
        sns.heatmap(
            samples[frame_idx[fr]],
            xticklabels=organelles,
            yticklabels=lipids,
            cmap="coolwarm",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=False
        )
        ax.set_title(f"{title} — run {frame_idx[fr]+1}/{M}")

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=interval)
    anim.save(out_fname, writer="pillow")
    plt.close(fig)
    gc.collect()
    return out_fname


# execution of the analysis

if __name__ == "__main__":

    # load and clean the data
    df = pd.read_csv("/Users/pellegri/Desktop/Year 2025/Tensor Lipid Transport /SummaryData.csv")
    df = df[df["UV"] == 1].dropna(subset=['Lipid', 'Organelle', 'Time', 'Mean'])  # keep only UV=1
    # keep only time =4 or >=30  
    df = df[(df["Time"] == 4) | (df["Time"] >= 30)].copy()

    # remove kb27 
    df = df[~df['Lipid'].isin(['KB27'])]
    # keep only 5 valid times 
    valid_times = [4, 30, 60, 120, 240]
    df = df[df['Time'].isin(valid_times)]
    df["Lipid"] = df["Lipid"].replace(LIPID_MAPPING)


    # count unique values for each mode
    lipids = sorted(df["Lipid"].unique())
    organelles = sorted(df["Organelle"].unique())
    time_steps = sorted(df["Time"].unique())

    lipid_to_idx = {lipid: i for i, lipid in enumerate(lipids)}
    organelle_to_idx = {organelle: i for i, organelle in enumerate(organelles)}
    time_to_idx = {time_step: i for i, time_step in enumerate(time_steps)}

    df["lipid_idx"] = df["Lipid"].map(lipid_to_idx)
    df["organelle_idx"] = df["Organelle"].map(organelle_to_idx)
    df["time_idx"] = df["Time"].map(time_to_idx)

    I, J, K = len(lipids), len(organelles), len(time_steps)  # size of the tensor

    # store the data inside a tensor

    tensor = np.zeros((I, J, K), dtype=np.float32)  # initialize to zero

    for _, row in df.iterrows():
        i, j, k = int(row['lipid_idx']), int(row['organelle_idx']), int(row['time_idx'])
        tensor[i, j, k] = row['Mean']

    # normalize each lipid with respect to frobenius norm, to make them comparable
    for i in range(I):
        slice_i = tensor[i, :, :]
    
        frob_norm = np.linalg.norm(slice_i, ord='fro')
    
        # scaling
        if frob_norm > 0:
            tensor[i, :, :] = slice_i / frob_norm


    # each entry of the tensor represents the normalized concentration of a certain lipid in a certain organelle at a certain time
    # print("Tensor shape (lipids x organeles x time):", tensor.shape)




    # compute SVD for the flattening of each mode 
    factors = [] 
    for mode in range(3): 
        unfolding = flattening(tensor, mode) # flatten along the mode 
        U, S, Vt = np.linalg.svd(unfolding, full_matrices=False) # svd of flattening 
        factors.append(U.astype(np.float32)) # store factor matrix U 


    U0, U1, U2 = factors 
    r0, r1, r2 = [U.shape[1] for U in factors] 
    print("Factor matrix U0 shape (I x r0):", U0.shape) 
    print("Factor matrix U1 shape (J x r1):", U1.shape) 
    print("Factor matrix U2 shape (K x r2):", U2.shape) 

    # compute the core tensor, via multilinear multiplication

    # mode 0 projection 
    A0 = np.tensordot(tensor, U0, axes=(0,0)) # shape: (J, K, r0) 
    A0 = np.transpose(A0, (2,0,1)) # shape: (r0, J, K) 
    # mode 1 projection 
    A1 = np.tensordot(A0, U1, axes=(1,0)) # shape: (r0, K, r1) 
    A1 = np.transpose(A1, (0,2,1)) # shape: (r0, r1, K) 
    # mode 2 projection 
    core = np.tensordot(A1, U2, axes=(2,0)) # shape: (r0, r1, r2) 
    print("Core tensor shape (r0 x r1 x r2):", core.shape)

    # check error of the approximation
    # contract core tensor with U0 along mode 0 
    temp = np.tensordot(core, U0, axes=(0,1)) # shape: (r1, r2, I) 
    temp = np.transpose(temp, (2,0,1)) # shape: (I, r1, r2) 
    # contract core tensor with U1 along mode 1 
    temp2 = np.tensordot(temp, U1, axes=(1,1)) # shape: (I, r2, J) 
    temp2 = np.transpose(temp2, (0,2,1)) # shape: (I, J, r2) 
    # contract core tensor with U2 along mode 2 
    reconstructed = np.tensordot(temp2, U2, axes=(2,1)) # shape: (I, J, K)

    explained_var = 1 - linalg.norm(tensor - reconstructed)/linalg.norm(tensor) # frobenius norm
    print("Explained variance:", explained_var) # if this is close to 1 the approximation is good

    number=1
    cluster_lipids(U0, core, number, lipids=lipids)
    temporal_strength_heatmap(U0, U1, U2, core, number, lipids=lipids, organelles=organelles)
    



    # truncated hosvd
    factors = [] 

    truncation_ranks = [4, 3, 3]

    for mode in range(3):
        unfolding = flattening(tensor, mode)
        U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)
        U_trunc = U[:, :truncation_ranks[mode]]
        factors.append(U_trunc.astype(np.float32))


    U0, U1, U2 = factors 
    r0, r1, r2 = [U.shape[1] for U in factors] 
    print("Factor matrix U0 shape (I x r0):", U0.shape) 
    print("Factor matrix U1 shape (J x r1):", U1.shape) 
    print("Factor matrix U2 shape (K x r2):", U2.shape) 

    # compute the core tensor, via multilinear multiplication

    # mode 0 projection 
    A0 = np.tensordot(tensor, U0, axes=(0,0)) # shape: (J, K, r0) 
    A0 = np.transpose(A0, (2,0,1)) # shape: (r0, J, K) 
    # mode 1 projection 
    A1 = np.tensordot(A0, U1, axes=(1,0)) # shape: (r0, K, r1) 
    A1 = np.transpose(A1, (0,2,1)) # shape: (r0, r1, K) 
    # mode 2 projection 
    core = np.tensordot(A1, U2, axes=(2,0)) # shape: (r0, r1, r2) 
    print("Core tensor shape (r0 x r1 x r2):", core.shape)

    # check error of approximation (i.e. the explained variance)
    # contract core tensor with U0 along mode-0 
    temp = np.tensordot(core, U0, axes=(0,1)) # shape: (r1, r2, I) 
    temp = np.transpose(temp, (2,0,1)) # shape: (I, r1, r2) 
    # contract core tensor with U1 along mode-1 
    temp2 = np.tensordot(temp, U1, axes=(1,1)) # shape: (I, r2, J) 
    temp2 = np.transpose(temp2, (0,2,1)) # shape: (I, J, r2) 
    # contract core tensor with U2 along mode-2 
    reconstructed = np.tensordot(temp2, U2, axes=(2,1)) # shape: (I, J, K)

    explained_var = 1 - linalg.norm(tensor - reconstructed)/linalg.norm(tensor) # frobenius norm
    print("Explained variance:", explained_var)

    number=2
    cluster_lipids(U0, core, number, lipids=lipids)
    temporal_strength_heatmap(U0, U1, U2, core, number, lipids=lipids, organelles=organelles)


    out_dir = "."  
    M_runs = 200
    bootstrap_seed = 123
    store_u0 = True
    ranks_none = None       


    X, STD, lipids_list, organelles_list, times_list = build_tensor_and_std_from_df(df, value_col='Mean')

    # run bootstrap
    res_full = bootstrap_hosvd(
        X, STD, M_runs=M_runs, ranks=ranks_none,
        what='strengths', seed=bootstrap_seed,
        verbose=True,
        store_u0=store_u0
    )

    #  run truncated bootstrap
    res_trunc = bootstrap_hosvd(
        X, STD, M_runs=M_runs, ranks=(4,3,3),
        what='strengths', seed=bootstrap_seed,
        verbose=True,
        store_u0=store_u0
    )


    fn1 = plot_strengths_heatmap_CI(res_full, lipids_list, organelles_list, title_prefix="Full_HOSVD", out_dir=out_dir)
    print("Saved CI heatmap:", fn1)
    fn2 = plot_strengths_heatmap_CI(res_trunc, lipids_list, organelles_list, title_prefix="Trunc_HOSVD", out_dir=out_dir)
    print("Saved CI heatmap:", fn2)

    
    try:
        gif_path = animate_bootstrap_strengths(res_full, lipids_list, organelles_list,
                                               title="Full HOSVD",
                                               out_fname=os.path.join(out_dir, "bootstrap_full.gif"),
                                               max_frames=60, interval=200)
        print("Saved animation to", gif_path)
    except Exception as e:
        print("Animation failed:", e)