# CP decomposition of a tensor
# Pipeline for: Tensor analysis for lipid transport 
# MPI-CBG, September 2025 


# import libraries 
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import CPTensor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import math

# mapping of the names of the 8 lipids 
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

# set tensorly backend to numpy 
tl.set_backend('numpy')


# class for consensus-based approach, used for rank selection and to avoid convergence to local minima
class ConsensusCP:
    def __init__(self, tensor, mask, max_rank=None, n_runs=15, variance_threshold=0.90,
                 cophenetic_threshold=0.80, random_state=42, lof_n_neighbors=20):
       
        self.tensor = tensor.astype(float)
        self.mask = mask.astype(int) # use the mask for missing values 
        self.dims = tensor.shape
        self.n_modes = len(self.dims)
        self.n_runs = int(n_runs)
        self.variance_threshold = float(variance_threshold)
        self.cophenetic_threshold = float(cophenetic_threshold)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
        self.lof_n_neighbors = int(lof_n_neighbors)

        
        self.max_rank = max_rank if max_rank is not None else 23

        self.results = {}           
        self.consensus_factors = {} # consensus factor list per rank

    
    def _extract_weights_and_factors(self, kruskal_like):
        eps = 1e-12 # tolerance
        if isinstance(kruskal_like, tuple) and len(kruskal_like) == 2:
            weights, factors = kruskal_like
            weights = np.asarray(weights).astype(float)
            return weights, [np.asarray(f).astype(float) for f in factors]
        if isinstance(kruskal_like, CPTensor):
            weights, factors = kruskal_like
            weights = np.asarray(weights).astype(float)
            return weights, [np.asarray(f).astype(float) for f in factors]
        # list of factor matrices
        if isinstance(kruskal_like, list):
            factors = [np.asarray(f).astype(float) for f in kruskal_like]
            R = factors[0].shape[1]
            weights = np.ones(R, dtype=float)
            return weights, factors
        raise ValueError("Error: unknown input!")

    def _compute_explained_variance(self, kruskal_like):
        try:
            weights, factors = self._extract_weights_and_factors(kruskal_like)
            reconstructed = tl.cp_to_tensor((weights, factors))

            mask = self.mask
            ss_res = np.sum(((self.tensor - reconstructed) * mask) ** 2)
            ss_tot = np.sum((self.tensor * mask) ** 2)

            if ss_tot == 0:
                return 0.0
            ev = 1.0 - ss_res / ss_tot

            # print, for diagnostic
            # print("ss_res_all   =", np.sum((self.tensor - reconstructed)**2))
            # print("ss_res_masked=", np.sum(((self.tensor - reconstructed)*self.mask)**2))
            # print("fraction used =", np.sum(self.mask) / self.tensor.size)

            return float(max(0.0, min(1.0, ev)))
        except Exception as e:
            print("Warning in _compute_explained_variance:", e)
            return 0.0


    def _cluster_columns(self, factors_per_mode, rank):
        mode_dims = [factors_per_mode[mode][0].shape[0] for mode in range(self.n_modes)]
        splits = np.cumsum(mode_dims)[:-1]
        mode_dim = int(np.sum(mode_dims))
        # stack columns from all runs, after normalization
        normed_factors_list = []
        for run_factors in zip(*[factors_per_mode[mode] for mode in range(self.n_modes)]):
            normed_modes = []
            for F in run_factors:
                norms = np.linalg.norm(F, axis=0)

                norms[norms == 0] = 1.0
                F_norm = F / norms
                normed_modes.append(F_norm)
            normed_factors_list.append(np.vstack(normed_modes))
        all_columns = np.hstack(normed_factors_list)


        # if the number of columns is less than two
        if all_columns.shape[1] < 2:
            print("Too few columns to cluster; returning random consensus.")
            random_consensus = np.column_stack([np.random.RandomState(self.random_state).rand(mode_dim)
                                                for _ in range(rank)])
            return np.split(random_consensus, splits, axis=0), [np.nan] * self.n_modes

        # LOF outlier removal
        n_samples = all_columns.shape[1]
        n_neighbors = min(self.lof_n_neighbors, max(2, n_samples - 1))
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            lof_labels = lof.fit_predict(all_columns.T)  
            inlier_mask = lof_labels == 1
            columns_clean = all_columns[:, inlier_mask]
            if columns_clean.shape[1] < rank:
                columns_clean = all_columns
        except Exception as e:
            # LOF failed 
            print("LOF failed. Error:", e)
            columns_clean = all_columns

        # k-means clustering on the columns 
        n_clusters = min(rank, columns_clean.shape[1])
        rng = np.random.RandomState(self.random_state)
        if n_clusters <= 1:
            labels = np.zeros(columns_clean.shape[1], dtype=int)
            silhouettes = [np.nan] * self.n_modes
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(columns_clean.T)
            # silhouette only defined if number of samples > n_clusters and n_clusters >= 2
            silhouettes = [np.nan] * self.n_modes
            try:
                if columns_clean.shape[1] > n_clusters:
                    silhouettes = [silhouette_score(block.T, labels)
                                   for block in np.split(columns_clean, splits, axis=0)]
            except Exception as e:
                silhouettes = [np.nan] * self.n_modes

        # median per cluster becomes consensus columns
        consensus_cols = []
        for cid in range(n_clusters):
            members = columns_clean[:, labels == cid]
            if members.shape[1] == 0:
                rnd = rng.rand(mode_dim)
                consensus_cols.append(rnd)
            else:
                consensus_cols.append(np.median(members, axis=1))

        consensus = np.column_stack(consensus_cols)

        # if fewer consensus columns than rank, fill with random
        while consensus.shape[1] < rank:
            consensus = np.column_stack([consensus, rng.rand(mode_dim)])

        # ensure exactly rank columns
        consensus = consensus[:, :rank]

        return (np.split(consensus, splits, axis=0),
                [float(s) if not np.isnan(s) else np.nan for s in silhouettes])

    def _compute_cophenetic(self, factors_per_mode, rank):
        try:

            if rank < 2: # for R = 1, a unique cluster: trivial
                return float('nan')
            n_runs = len(factors_per_mode[0])
            consensus_matrix = None
            for run_factors in zip(*[factors_per_mode[mode] for mode in range(self.n_modes)]):
                flat = np.vstack([F / np.maximum(np.linalg.norm(F, axis=0), 1e-12) for F in run_factors])
                assignment = np.argmax(flat, axis=1)
                connectivity = (assignment[:, None] == assignment[None, :]).astype(float)
                consensus_matrix = connectivity if consensus_matrix is None else consensus_matrix + connectivity
            consensus_matrix = consensus_matrix / n_runs
            dist = squareform(1.0 - consensus_matrix, checks=False)
            if np.ptp(dist) == 0:
                return float('nan')
            linkage_matrix = linkage(dist, method='average')
            coph_corr, _ = cophenet(linkage_matrix, dist)
            return float(coph_corr)
        except Exception as e:
            print("Warning in _compute_cophenetic:", e)
            return 0.0



    def run_consensus_analysis(self):
        print(f"Consensus CP: tensor shape={self.dims}, max_rank={self.max_rank}, n_runs={self.n_runs}")
        for rank in range(1, int(self.max_rank) + 1):
            all_factors_per_mode = {mode: [] for mode in range(self.n_modes)}
            variances = []
            run_reprs = []  

            # multiple random runs for this rank
            for run in range(self.n_runs):
                rs = self.random_state + run
                try:
                    # kruskal = parafac(
                        # self.tensor,
                        # mask=self.mask,
                        # rank=rank,
                        # n_iter_max=2000,
                        # tol=1e-6,
                        # init='random',
                        # random_state=rs,
                        # normalize_factors=True, # normalize the factors automatically
                    # )
                    kruskal = non_negative_parafac(
                        self.tensor,
                        mask=self.mask,
                        rank=rank,
                        n_iter_max=2000,
                        tol=1e-6,
                        init='random',
                        random_state=rs,
                        normalize_factors=True, # normalize the factors automatically
                    )
                    weights, factors = self._extract_weights_and_factors(kruskal)
                    var_exp = self._compute_explained_variance((weights, factors))
                    variances.append(var_exp)
                    run_reprs.append((weights, factors))
                    for mode in range(self.n_modes):
                        all_factors_per_mode[mode].append(factors[mode])
                except Exception as e:
                    print(f"Rank {rank} run {run} failed: {e}")
                    continue

            if len(variances) == 0:
                print(f"Rank {rank}: all runs failed, skipping.")
                continue

            # calculate thhe consensus per mode and collect silhouette scores
            consensus, silhouettes = self._cluster_columns(all_factors_per_mode, rank)

            # compute cophenetic correlation across consensus
            coph = self._compute_cophenetic(all_factors_per_mode, rank)

            # normalize consensus columns and compute weights 
            eps = 1e-12
            consensus_normed = [np.array(m, dtype=float).copy() for m in consensus]
            R = rank
            weights = np.ones(R, dtype=float)
            for r in range(R):
                prod = 1.0
                for mode in range(self.n_modes):
                    col = consensus_normed[mode][:, r]
                    nrm = np.linalg.norm(col)
                    if nrm < eps:
                        nrm = eps
                    consensus_normed[mode][:, r] = col / nrm
                    prod *= nrm
                weights[r] = prod

            # final CP decomposition using consensus initialization
            try:
                init_cpt = (weights, consensus_normed)  
                # final_kruskal = parafac(
                    # self.tensor, rank=rank, n_iter_max=3000,
                    # init=init_cpt, tol=1e-8, random_state=self.random_state, normalize_factors=True
                # ) 
                final_kruskal = non_negative_parafac(
                    self.tensor, mask=self.mask, rank=rank, n_iter_max=3000,
                    init=init_cpt, tol=1e-8, random_state=self.random_state, normalize_factors=True
                )
                final_weights, final_factors = self._extract_weights_and_factors(final_kruskal)
                final_var = self._compute_explained_variance((final_weights, final_factors))

                # print, just for diagnostic
                print("   [Final model] error on observed:",
                    np.sum(((self.tensor - tl.cp_to_tensor((final_weights, final_factors))) * self.mask)**2))

                print("   [Final model] error on unobserved:",
                    np.sum(((self.tensor - tl.cp_to_tensor((final_weights, final_factors))) * (1 - self.mask))**2))

            
            except Exception as e:
                print(f"Final parafac with consensus init failed for rank {rank}: {e}")
                # if failure, pick best run of earlier random runs
                idx = int(np.argmax(variances))
                final_weights, final_factors = run_reprs[idx]
                final_var = variances[idx]

            # save results for this rank
            self.results[rank] = {
                'variance': float(final_var),
                'cophenetic': float(coph),
                'silhouettes': silhouettes,
                'consensus_weights': weights,
                'consensus_factors': consensus,         
                'consensus_normed': consensus_normed,    
                'final_weights': final_weights,
                'final_factors': final_factors
            }
            self.consensus_factors[rank] = consensus

            sil_mean = np.nanmean(silhouettes) if np.any(~np.isnan(silhouettes)) else float('nan')
            print(f"Rank {rank}: Variance={final_var:.4f}, Cophenetic={coph:.4f}, Silhouette(avg)={sil_mean if not np.isnan(sil_mean) else 'nan'}")




    def select_optimal_rank(self):
        if not self.results:
            raise RuntimeError("No results available.")

        def sil_avg(res):
            sils = res.get('silhouettes', [])
            # handle NaNs
            if isinstance(sils, (list, tuple)) and len(sils) > 0 and np.any(~np.isnan(sils)):
                return float(np.nanmean(sils))
            return float('nan')

        # ranks that satisfy thresholds
        valid = []
        for r, res in self.results.items():
            if res['variance'] >= self.variance_threshold and res['cophenetic'] >= self.cophenetic_threshold:
                valid.append((r, sil_avg(res), res['variance']))

        if valid:
            # pick max silhouette
            valid.sort(key=lambda t: (
                -(t[1] if not np.isnan(t[1]) else -np.inf),   # higher silhouette
                -t[2],                                        # higher variance
                t[0]                                         # lower rank
            ))
            best_rank = int(valid[0][0])
            print(f"Selected optimal rank (meets thresholds, silhouette peak): {best_rank}")
            return best_rank

        
        all_ranks = []
        for r, res in self.results.items():
            all_ranks.append((r, sil_avg(res), res['variance']))

        if any(not np.isnan(s) for _, s, _ in all_ranks):
            all_ranks.sort(key=lambda t: (-(t[1] if not np.isnan(t[1]) else -np.inf), -t[2], t[0]))
            best_rank = int(all_ranks[0][0])
            print(f"No rank meets thresholds; fallback to global silhouette peak at rank {best_rank}")
            return best_rank
        # last: highest variance
        best_rank = max(self.results.keys(), key=lambda r: self.results[r]['variance'])
        print("No rank meets thresholds and all silhouettes are NaN; selecting rank with highest variance:", best_rank)
        return best_rank
    

    def get_factors(self, rank=None):
        if rank is None:
            rank = self.select_optimal_rank()
        res = self.results.get(rank)
        if res is None:
            raise KeyError(f"No results stored for rank {rank}.")
        return res['final_weights'], res['final_factors']



def prepare_tensor(df):
    # some lipids have missing time measurements
    # keep only time =4 or >=30 
    # df = df[(df["Time"] == 4) | (df["Time"] >= 30)].copy()
    # alternative: to handle the missing values, i could do the CP but using a mask, to maximize the likelihood only wrt the observed values
    # and then at the end the resulting factors come from optimization of just the observed entries, not of all the tensor filled artificially

    lipids = sorted(df["Lipid"].unique())
    organelles = sorted(df["Organelle"].unique())
    times = sorted(df["Time"].unique())

    I, J, K = len(lipids), len(organelles), len(times)

    tensor = np.zeros((I, J, K), dtype=np.float32)
    mask = np.zeros((I, J, K), dtype=int)

    lipid_idx = {lipid: i for i, lipid in enumerate(lipids)}
    organelle_idx = {org: j for j, org in enumerate(organelles)}
    time_idx = {t: k for k, t in enumerate(times)}

    for _, row in df.iterrows():
        i = lipid_idx[row['Lipid']]
        j = organelle_idx[row['Organelle']]
        k = time_idx[row['Time']]

        val = row['Mean']
        if pd.isna(val):
            continue  
        tensor[i, j, k] = float(val)
        mask[i, j, k] = 1

    for i in range(I):
        slice_i = tensor[i, :, :]
        mask_i = mask[i, :, :]

        frob_norm = np.sqrt(np.sum((slice_i * mask_i) ** 2))

        if frob_norm > 0:
            tensor[i, :, :] = (slice_i / frob_norm) * mask_i


    return tensor, mask, lipids, organelles, times, I, J, K


def plot_rank_metrics(results, rank_min=1, rank_max=None, mode_index=None,
                      var_thr=0.90, coph_thr=0.80):

    if not results:
        raise RuntimeError("No results to plot.")

    # restrict to available ranks
    all_ranks = sorted(results.keys())
    if rank_max is None:
        rank_max = max(all_ranks)

    rows = []
    for r in all_ranks:
        if r < rank_min or r > rank_max:
            continue
        res = results[r]
        sils = res.get("silhouettes", [])
        if sils:
            if mode_index is None:
                sil = np.nanmean(sils) if np.any(~np.isnan(sils)) else np.nan
            elif mode_index < len(sils):
                sil = sils[mode_index]
            else:
                sil = np.nan
        else:
            sil = np.nan

        rows.append((r, res["variance"], res["cophenetic"], sil))

    if not rows:
        raise ValueError("No results in the requested rank range.")

    df = pd.DataFrame(rows, columns=["rank","variance","cophenetic","silhouette"])

    # plotting

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1, color2, color3 = "tab:blue", "tab:green", "tab:red"
    ax1.plot(df["rank"], df["variance"], "o-", color=color1, label="Explained Variance")
    ax1.plot(df["rank"], df["cophenetic"], "s-", color=color2, label="Cophenetic Corr.")
    ax1.set_xlabel("Rank")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(df["rank"], df["silhouette"], "d-", color=color3, label="Silhouette")
    ax2.set_ylim(0, 1.05)


    # merge legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    plt.savefig("consensus_cp_metrics.svg", format="svg", bbox_inches="tight")
    plt.close()

    return df


from scipy.optimize import linear_sum_assignment


def random_missing_mask(mask, percentage, rng):
    """
    Randomly hide `percentage` % of the truly observed entries of the tensor.

    mask       : binary mask of the truly observed entries (1 = observed, 0 = missing in the raw data)
    percentage : percentage (e.g. 10 for 10%) of the observed entries to turn into missing values
    rng        : numpy Generator, for reproducibility

    Returns
      mask_missing : binary mask used for the fit (1 = kept, 0 = missing)
      holdout_mask : binary mask of the entries that were hidden on purpose
    """
    observed_idx = np.flatnonzero(mask.ravel() == 1)
    n_remove = int(round(percentage / 100.0 * observed_idx.size))
    removed_idx = rng.choice(observed_idx, size=n_remove, replace=False)

    mask_missing = mask.copy().astype(int)
    mask_missing.ravel()[removed_idx] = 0
    holdout_mask = (mask - mask_missing).astype(int)
    return mask_missing, holdout_mask


def relative_frobenius_error(tensor, reconstructed, mask):
    """
    Relative Frobenius norm of the error, computed only on the entries where mask == 1:
        || mask * (T - T_hat) ||_F / || mask * T ||_F
    """
    num = np.linalg.norm((tensor - reconstructed) * mask)
    den = np.linalg.norm(tensor * mask)
    return float(num / den) if den > 0 else float("nan")


def align_components(weights, factors, ref_factors):
    """
    CP components come out in arbitrary order. Reorder them so that component r
    matches component r of a reference decomposition (here: the one on the full tensor),
    so that the bar plots are comparable across the different missing percentages.
    Matching = Hungarian algorithm on the sum over modes of the cosine similarities.
    """
    R = factors[0].shape[1]
    sim = np.zeros((R, R))
    for F, G in zip(ref_factors, factors):
        Fn = F / np.maximum(np.linalg.norm(F, axis=0), 1e-12)
        Gn = G / np.maximum(np.linalg.norm(G, axis=0), 1e-12)
        sim += np.abs(Fn.T @ Gn)
    _, order = linear_sum_assignment(-sim)
    return np.asarray(weights)[order], [F[:, order] for F in factors]


def plot_factor_bars(weights, factors, lipids, organelles, times, filename, title=None):
    """Bar plots of the CP factors: one row per component, columns = lipids / organelles / times."""
    A, B, C = factors
    R = A.shape[1]
    fig, axes = plt.subplots(nrows=R, ncols=3, figsize=(15, 4 * R), squeeze=False)

    for r in range(R):
        ax_lipid = axes[r, 0]
        ax_lipid.bar(range(len(lipids)), A[:, r])
        ax_lipid.set_xticks(range(len(lipids)))
        ax_lipid.set_xticklabels(lipids, rotation=90)
        ax_lipid.set_ylim([-0.2, 1.1])

        ax_organelle = axes[r, 1]
        ax_organelle.bar(range(len(organelles)), B[:, r])
        ax_organelle.set_xticks(range(len(organelles)))
        ax_organelle.set_xticklabels(organelles, rotation=45)
        ax_organelle.set_ylim([-0.2, 1.1])

        ax_time = axes[r, 2]
        ax_time.bar(range(len(times)), C[:, r])
        ax_time.set_xticks(range(len(times)))
        ax_time.set_xticklabels(times, rotation=45)
        ax_time.set_ylim([-0.2, 1.1])

        axes[r, 0].text(-0.08, 1.04, 'ABCDEFGHIJ'[r], transform=axes[r, 0].transAxes,
                        fontsize=20, fontweight='bold', va='bottom', ha='right')
        axes[r, 1].set_title(f"weight $\\lambda_{r+1}$ = {weights[r]:.3f}", fontsize=12)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_missing_error_curve(summary_df, filename="relative_error_vs_missing.pdf"):
    """Relative Frobenius error (in %) as a function of the percentage of missing values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["missing_pct"], 100 * summary_df["rel_error_observed"], "o-", lw=2.5,
            color="tab:blue", label="all truly observed entries")
    ax.plot(summary_df["missing_pct"], 100 * summary_df["rel_error_fit"], "s--",
            color="tab:green", label="entries used in the fit")
    ax.plot(summary_df["missing_pct"], 100 * summary_df["rel_error_holdout"], "d--",
            color="tab:red", label="hidden entries only")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Relative Frobenius error (%)")
    ax.set_xticks(summary_df["missing_pct"])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Non-negative CP: relative error vs. missing values")
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


# execution

if __name__ == "__main__":
    df = pd.read_csv("/Users/pellegri/Desktop/Year 2025/Tensor Lipid Transport /SummaryData.csv")
    df = df[df["UV"] == 1] # keeping only data for UV = 1

    missing_rows = df[pd.isna(df["Mean"])]

    # remove kb27 
    df = df[~df['Lipid'].isin(['KB27'])]
    # keep only these times 
    valid_times = [4, 30, 60, 120, 240]
    df = df[df['Time'].isin(valid_times)]
    df["Lipid"] = df["Lipid"].replace(LIPID_MAPPING)
    

    tensor, mask, lipids, organelles, times, I, J, K = prepare_tensor(df)
    consensus_cp = ConsensusCP(tensor,mask, max_rank=6, n_runs=20,
                               variance_threshold=0.90, cophenetic_threshold=0.80,
                               random_state=42)
    consensus_cp.run_consensus_analysis()
    # optimal_rank = consensus_cp.select_optimal_rank()
    optimal_rank=3
    weights, factors = consensus_cp.get_factors(optimal_rank)
    reconstructed_full = tl.cp_to_tensor((weights, factors))


    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import numpy as np

    A, B, C = factors
    R = optimal_rank

    # creates figure with R rows 3 columns
    fig, axes = plt.subplots(nrows=R, ncols=3, figsize=(15, 4 * R))

    for r in range(R):
        # lipid mode
        ax_lipid = axes[r, 0]
        ax_lipid.bar(range(len(lipids)), A[:, r])
        ax_lipid.set_xticks(range(len(lipids)))
        ax_lipid.set_xticklabels(lipids, rotation=90)
        ax_lipid.set_ylim([-0.2, 1.1]) 
        
        # organelle mode
        ax_organelle = axes[r, 1]
        ax_organelle.bar(range(len(organelles)), B[:, r])
        ax_organelle.set_xticks(range(len(organelles)))
        ax_organelle.set_xticklabels(organelles, rotation=45)
        ax_organelle.set_ylim([-0.2, 1.1])
        
        # time mode
        ax_time = axes[r, 2]
        ax_time.bar(range(len(times)), C[:, r])
        ax_time.set_xticks(range(len(times)))
        ax_time.set_xticklabels(times, rotation=45)
        ax_time.set_ylim([-0.2, 1.1])

    # panel letters
    for r in range(R):
        axes[r, 0].text(-0.08, 1.04, 'ABCDEFGHIJ'[r], transform=axes[r, 0].transAxes,
                        fontsize=20, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout()
    plt.savefig(f"Output of CP with rank {R}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

   
    full_weights, full_factors = weights, factors
    for mat, labels, name in zip(full_factors, (lipids, organelles, times), ("lipid", "organelle", "time")):
        pd.DataFrame(mat, index=labels, columns=[f"Comp{r+1}" for r in range(optimal_rank)]
                     ).to_csv(f"{name}_factors.csv")

    # rank metrics of the full-data consensus analysis
    metrics_df = plot_rank_metrics(
        consensus_cp.results,
        rank_min=1, rank_max=15,
        mode_index=None,  # average of silhouette score along 3 modes
        var_thr=0.90, coph_thr=0.80
    )

    print("Number of missing values in the original tensor:", int(np.sum(mask == 0)))
    observed_values = tensor[mask == 1]
    val_min, val_max = observed_values.min(), observed_values.max()
    print(f"[INFO] Normalized tensor - min entry: {val_min:.6f}, max entry: {val_max:.6f}")


    # try percentages of missing values
    # for each one
    #   1. randomly hide that percentage of the truly observed entries
    #   2. run masked consensus non-negative CP
    #   3. plot the factor bars
    #   4. compute the relative frobenius error on the truly observed entries
    MISSING_PERCENTAGES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    MISSING_SEED = 0          
    rng = np.random.default_rng(MISSING_SEED)

    # reference point: 0% artificial missing values
    summary_rows = [{
        "missing_pct": 0,
        "n_hidden": 0,
        "rel_error_observed": relative_frobenius_error(tensor, reconstructed_full, mask),
        "rel_error_fit": relative_frobenius_error(tensor, reconstructed_full, mask),
        "rel_error_holdout": np.nan,
    }]

    for pct in MISSING_PERCENTAGES:

        mask_missing, holdout_mask = random_missing_mask(mask, pct, rng)
        tensor_missing = tensor * mask_missing      # hidden entries set to 0 (ignored through the mask)
        print(f"Hidden entries: {int(holdout_mask.sum())} of {int(mask.sum())} observed "
              f"(total missing now: {int(np.sum(mask_missing == 0))} of {tensor.size})")

        cp_missing = ConsensusCP(tensor_missing, mask_missing, max_rank=6, n_runs=20,
                                 variance_threshold=0.90, cophenetic_threshold=0.80,
                                 random_state=42)
        cp_missing.run_consensus_analysis()
        w_miss, f_miss = cp_missing.get_factors(optimal_rank)

        # same component order as the full-data decomposition
        w_miss, f_miss = align_components(w_miss, f_miss, full_factors)
        reconstructed_missing = tl.cp_to_tensor((w_miss, f_miss))

        # relative frobenius error, only on entries whose true value is known
        err_observed = relative_frobenius_error(tensor, reconstructed_missing, mask)          # all truly observed
        err_fit = relative_frobenius_error(tensor, reconstructed_missing, mask_missing)       # entries used in the fit
        err_holdout = relative_frobenius_error(tensor, reconstructed_missing, holdout_mask)   # hidden entries only
        print(f"Relative Frobenius error - all observed: {100*err_observed:.2f}% | "
              f"fit entries: {100*err_fit:.2f}% | hidden entries: {100*err_holdout:.2f}%")

        summary_rows.append({
            "missing_pct": pct,
            "n_hidden": int(holdout_mask.sum()),
            "rel_error_observed": err_observed,
            "rel_error_fit": err_fit,
            "rel_error_holdout": err_holdout,
        })

        # factor bar plots
        plot_factor_bars(
            w_miss, f_miss, lipids, organelles, times,
            filename=f"Output of CP with rank {optimal_rank} and {pct}pct missing values.pdf",
            title=f"{pct}% missing values - relative error on hidden entries: {100*err_holdout:.2f}%"
        )

        # factor matrices
        for mat, labels, name in zip(f_miss, (lipids, organelles, times), ("lipid", "organelle", "time")):
            pd.DataFrame(mat, index=labels, columns=[f"Comp{r+1}" for r in range(optimal_rank)]
                         ).to_csv(f"{name}_factors_{pct}pct_missing.csv")

    # summary over all percentages
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("relative_error_vs_missing.csv", index=False)
    plot_missing_error_curve(summary_df, "relative_error_vs_missing.pdf")

    
    print(summary_df.assign(
        rel_error_observed=lambda d: (100 * d.rel_error_observed).round(2),
        rel_error_fit=lambda d: (100 * d.rel_error_fit).round(2),
        rel_error_holdout=lambda d: (100 * d.rel_error_holdout).round(2),
    ).to_string(index=False))
    print("(errors in %)")