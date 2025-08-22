# Validation Protocol for Hybrid vs. Classic Mini-Delphi

## Background: Validity Concepts

- **Face validity**: Refers to whether the items and dimensions appear, on the surface, to measure what they are intended to measure. It is a subjective judgment made by experts or users about the clarity, fit, and usability of the items.

- **Content validity**: Refers to whether the items and dimensions adequately represent the construct domain. It evaluates coverage (are all important aspects included?) and redundancy (are there unnecessary overlaps?). It is usually quantified using indices like **I-CVI** (item-level content validity index), **S-CVI/Ave** (scale-level average CVI), and **modified kappa** to adjust for chance agreement.

- **Non-inferiority**: In comparative studies, non-inferiority means showing that a new method (Hybrid Delphi) is **not worse than** an established method (Classic Delphi) by more than a small, pre-specified margin (δ). If the lower bound of the confidence interval for the difference Hybrid−Classic is greater than −δ, the new method can be declared **non-inferior**.

---

# Step-by-Step Protocol

## 0) Pre-registration
- Primary outcome: Non-inferiority of expert-rated quality (Composite of Clarity, Fit, Redundancy) for Hybrid vs. Classic.  
- Margin (δ): e.g., 0.30 Likert units on a 1–5 scale.  
- Secondary outcomes: Content validity indices (I-CVI, S-CVI/Ave, κ*), rater agreement (ICC or AC1), convergence, efficiency.  
- Exclusion rules: Failed attention checks, incomplete forms.  
- Blinding: Independent panel blinded to taxonomy origin.  
- Analysis plan: Mixed-effects model or paired non-parametric tests.

## 1) Materials Preparation
- Item pool: Use same 40 items and construct scope.  
- Prompts & decision rules: Keep parallel across methods (Hybrid vs. Classic).  
- Data capture forms: Grouping matrices, rating sheets.  
- Decision rules: Retain if Fit ≥ 4, Clarity ≥ 4, IQR ≤ 1. Revise if marginal. Drop if Fit < 3.5 or redundancy flagged. Coverage safeguard: ≥2 items per dimension.

## 2) Classic Mini-Delphi (6 Experts)
- **Round 1:** Experts group items into dimensions, label, define. Moderator synthesizes.  
- **Round 2:** Experts revise based on synthesis. Moderator finalizes dimension set.  
- **Round 3:** Experts rate Fit, Clarity, Redundancy. Moderator applies decision rules.  
- **Logs:** Capture expert-minutes, dropout, consensus indices (Kendall’s W, IQR), stability (Jaccard).

## 3) Freeze Outcomes
- Hybrid outcome (from synthetic + human pipeline).  
- Classic outcome (from Step 2).  
- Mask provenance: “Taxonomy A” vs “Taxonomy B”. Shuffle items.

## 4) Independent, Blinded Expert Panel (N=12–20)
- Each rater evaluates both taxonomies, blinded and order-randomized.  
- **Item-level ratings:** Fit (1–5), Clarity (1–5), Redundancy (Y/N), Relevance (for CVI).  
- **Dimension-level ratings:** Coverage adequate (Y/N), Label clarity (1–5).  
- **Global:** Usability (1–5), Coherence (1–5).  
- Attention checks included.

## 5) Measures

### 5.1 Face Validity (Primary)
- Composite Quality = z(Fit) + z(Clarity) – z(Redundancy%).  
- Non-inferiority test: Hybrid−Classic ≥ −δ.

### 5.2 Content Validity
- **I-CVI:** Proportion of raters marking item relevance acceptable.  
- **S-CVI/Ave:** Average I-CVI across items.  
- **Modified kappa (κ*):** Adjusts I-CVI for chance agreement.

### 5.3 Rater Agreement
- **ICC(2,k):** For continuous ratings.  
- **Gwet’s AC1/AC2:** For binary judgments.

### 5.4 Convergence & Stability
- Kendall’s W and IQR by round.  
- Jaccard/ARI similarity between rounds.

### 5.5 Efficiency
- Expert-minutes, dropout, rounds, elapsed days.

## 6) Analysis

- **Primary test:** Mixed-effects model  
  ```  
  Composite ~ Method + (1|Rater) + (1|Item)  
  ```  
  Non-inferiority: lower 95% CI > −δ.  
- **Alternative:** Paired Wilcoxon with Hodges–Lehmann CI.  
- **Content validity:** Compute I-CVI, S-CVI/Ave, κ*. Compare Hybrid vs Classic with paired tests.  
- **Agreement:** ICC/AC1 with 95% CI.  
- **Convergence & efficiency:** Plot Kendall’s W, Jaccard; compare expert-minutes.

## 7) Reporting Package

**Tables:**  
1. Expert panels & burden.  
2. Primary non-inferiority results.  
3. Content validity indices.  
4. Rater agreement.  
5. Convergence & efficiency.  

**Figures:**  
- Line chart: consensus (Kendall’s W) across rounds.  
- Violin plot: Composite Quality Hybrid vs Classic.  
- Bar chart: S-CVI/Ave and κ*.  
- Sankey diagram (appendix): item movements.

---

# Practical Notes
- Classic mini-Delphi: 6 experts, 3 rounds.  
- Independent panel: Aim 12–20 raters (≥10 minimum).  
- Data prep: Tidy tables for items, ratings, global judgments, process logs.  
- Ethics: Consent, anonymization, compensation/acknowledgement.  
- Data sharing: De-identified aggregated data + code.

---
