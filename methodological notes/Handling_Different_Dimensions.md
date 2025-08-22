# Handling Different Dimensions in Hybrid vs Classic Delphi Comparisons

## Background

When comparing **Hybrid Delphi** and **Classic mini-Delphi**, the number, labels, and membership of dimensions may differ.  
This does **not** invalidate comparisons, but it requires **label-free, structure-based metrics** and transparent reporting.

---

## Step-by-Step Strategy

### 1. Label-Agnostic Comparisons
- Treat dimensions as **clusters** of items, not by their labels.  
- Ignore semantic names (“Anxiety”, “Stress”), focus on item membership.

### 2. Build an Overlap Matrix
- Let Hybrid clusters = H1...Hk, Classic clusters = C1...Cm.  
- Compute **overlap count matrix**:  
  \( M_{ij} = |H_i ∩ C_j| \)  
- Compute **Jaccard index**:  
  \( J_{ij} = |H_i ∩ C_j| / |H_i ∪ C_j| \)

### 3. Optimal Matching
- Use the **Hungarian algorithm** to align clusters based on maximum Jaccard overlap.  
- Cost matrix = 1 − Jaccard.  
- Produces best one-to-one mapping; unmatched clusters = **orphans**.

### 4. Use Structure Metrics (Permutation-Invariant)
- **Adjusted Rand Index (ARI)** on item assignments.  
- **Normalized Mutual Information (NMI)** (optional).  
- **Mean Jaccard** across matched pairs.

### 5. Item-Level Comparisons (Independent of Dimensions)
- Compute **Face validity** (Fit, Clarity, Redundancy) per item.  
- Compute **Content validity** (I-CVI, κ*) per item.  
- Aggregate across items per taxonomy.  
- Differences in dimension structures do not matter here.

### 6. Handling Splits and Merges
- After matching, report **spillover**: secondary overlaps (next-highest Jaccard).  
- Orphans: report separately (item count + best secondary match).

### 7. Fair Comparisons with Different Item Sets
- **As-proposed analysis (primary):** Each taxonomy judged on its own final set of items.  
- **Common-core sensitivity (secondary):** Restrict to items retained by both methods.

### 8. Dimension-Level Quality
- Compare coverage and label clarity on matched pairs.  
- Report orphans’ scores transparently.

### 9. Convergence & Stability (Process Metrics)
- Within-method stability: Kendall’s W and IQR(Fit) across rounds.  
- Structural stability: Jaccard/ARI between R1 → R2 within each method.  
- Visualize with **Sankey diagrams**.

### 10. Reporting Language
- *“Structures were compared using permutation-invariant indices (ARI, Jaccard). Clusters were aligned using optimal matching; unmatched clusters reported separately.”*  
- *“Face/content validity were item-level, thus unaffected by different labels.”*  
- *“Analyses were conducted both on full taxonomies (as-proposed) and on a common-core intersection (sensitivity).”*

---

## Outputs to Include

1. **Structure agreement (label-free):** ARI, Jaccard, NMI.  
2. **Overlap matrix:** counts and Jaccard values; matched pairs highlighted; orphans listed.  
3. **Face validity:** Composite quality per taxonomy; non-inferiority test.  
4. **Content validity:** I-CVI, S-CVI/Ave, κ* distributions.  
5. **Process:** Kendall’s W trajectories, stability indices, burden metrics.  
6. **Visuals:**  
   - Sankey (item flows across rounds).  
   - Heatmap of overlap matrix.  
   - Line plots of consensus/stability.

---

## Implementation Notes

- **Hungarian matching:** cost = 1 − Jaccard; pad matrix if clusters differ.  
- **Bootstrapped CIs:** resample items to get 95% CI for ARI/Jaccard.  
- **Non-inferiority tests:** apply to **expert-rated composites**, not structural metrics.

---
