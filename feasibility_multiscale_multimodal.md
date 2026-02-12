# Feasibility: Multi-Scale Multi-Modal Data for Fine-Grained Correspondences

## Reference: Multimodal Synthetic Data

The framework in `synthetic_multiscale_multimodal_data.ipynb` (multiscale repo) provides:

### Two modalities

- **S1 (driving signal)**  
  - High-resolution spatiotemporal signal.  
  - **Multiscale in frequency**: sum of sinusoids with different spatial wavenumbers `k` and temporal frequencies `omega` (large/slow, medium, small/fast).  
  - Optional: amplitude modulation (AM), chirp, and correlated (e.g. pink) noise.  
  - Grid: `(x, t)` 1D or `(x, y, t)` 2D.

- **S2 (coupled signal)**  
  - Lower resolution: coarser spatial grid `(x2, t2)` and/or `(x2, y2, t2)`.  
  - **Driven by S1** via a Kuramoto-like phase coupling: at each S2 spatial point, phases evolve as  
    `dθ_j/dt = ω'_j + K * mean_field(sin(ψ_i - θ_j))`,  
    where `ψ_i` are S1 component phases (interpolated to current time).  
  - So the coupling is **nonlinear and at the phase level**; strength is controlled by **K**.

### Existing "correspondence"

- **Spatial**: Each S2 grid point `(x2[i], y2[j])` is driven by S1 at the **nearest** high-res point `(x[i'], y[j'])` (nearest-neighbor in the reference implementation).  
- **Temporal**: S2 is evaluated only at `t2`; the ODE is integrated using S1 phases interpolated on the fine `t` grid.  
- **Ground truth**: Coupling strength **K** (and the fact that S1 drives S2).

So we already have **two resolutions** (S1 fine, S2 coarse) and an **implicit** fine–coarse correspondence (nearest neighbor). What we do **not** have yet is an **explicit multi-scale pyramid** and **fine-grained correspondence maps** suitable for learning.

---

## Goal: Multi-Scale Data for Learning Fine-Grained Correspondences

We want to extend this so that:

1. **Multiple scale levels** exist explicitly (e.g. level 0 = finest, 1 = medium, 2 = coarsest).  
2. **Fine-grained correspondence** is explicit: for each coarse cell (or point) we know exactly which fine-grid indices (spatial and temporal) it corresponds to (e.g. which fine (x, t) "belong" to that coarse cell).  
3. Data can be used to **learn** these correspondences (e.g. which fine S1 region corresponds to which coarse S2 cell, or vice versa), possibly in addition to predicting K.

---

## Proposed Multi-Scale Extension

### 1. Pyramid of grids

- **Level 0 (finest)**: `x_0`, `t_0` (and in 2D: `x_0`, `y_0`, `t_0`).  
- **Level 1**: `x_1`, `t_1` with e.g. 2× or 4× fewer points (regular downsampling or averaging).  
- **Level 2**: `x_2`, `t_2` again coarser (e.g. 2× or 4× fewer than level 1).

Ratios can be fixed (e.g. 4:2:1 in space and time) so that each coarse point corresponds to a **block** of fine indices.

### 2. What lives at each scale

- **S1**: Generated on the **finest grid only** (level 0).  
  - Coarser views of S1 can be obtained by **downsampling/averaging** (e.g. S1_level1 = average over 2×2×2 blocks in (x, y, t)).  
  - This gives a consistent multi-scale representation of the same underlying process.  

- **S2**: Generated on **one or several** coarse grids using the existing Kuramoto coupling:  
  - At each coarse spatial point, drive S2 with S1's phases at the **corresponding** fine point (or at the centroid of the block that this coarse point represents).  
  - So we can have S2 at level 1 and S2 at level 2, each with a well-defined link to S1 at level 0.

### 3. Explicit correspondence (fine-grained)

For **learning** correspondences, we store:

- **Coarse → fine (per level)**  
  - For each coarse index `(i_c, j_c)` at level `L`, the list (or slice) of fine indices `(i_f, j_f)` (and time) that belong to that coarse cell.  
  - Example (1D space): coarse cell `i_c` at level 1 might correspond to fine indices `[i_c * r : (i_c+1) * r]` with ratio `r`.

- **Fine → coarse**  
  - For each fine index, which coarse cell(s) it belongs to (for aggregation or loss weighting).

These can be stored as:

- Lists of slices (e.g. `fine_slices_by_coarse[i_c] = (slice(i_f_lo, i_f_hi), slice(j_f_lo, j_f_hi))`), or  
- Index arrays (e.g. `fine_to_coarse_spatial[i_f, j_f] = (i_c, j_c)`).

Same idea for **time**: coarse time step `k_c` corresponds to fine time indices `[k_c * r_t : (k_c+1) * r_t]`.

### 4. Feasibility

- **Generator**: The same `generate_base_signal` and Kuramoto ODE logic can be reused.  
  - S1 is generated once on the fine grid.  
  - For each coarse level, we define `x_coarse`, `t_coarse` and, for each coarse spatial point, the corresponding fine index (or block centroid) to take S1's phases from; then we run the same `solve_ivp` + sum of sines to get S2 at that level.  
- **Correspondence**: Pure bookkeeping: define downsampling ratios and compute index maps (coarse ↔ fine). No change to the physics.  
- **Storage**: Save in addition to `S1`, `S2`, `K`:  
  - `grids`: `(x_0, t_0, x_1, t_1, ..., x_L, t_L)` (and in 2D, y as well).  
  - `correspondence`: e.g. list of dicts per level, or arrays like `fine_to_coarse_spatial`, `coarse_to_fine_slices`.  

So **multi-scale data with fine-grained correspondences is feasible** within the existing multimodal synthetic framework by:

1. Generating S1 on a single fine grid.  
2. Defining a pyramid of coarse grids and, for each coarse point, which fine point (or block) drives S2.  
3. Generating S2 at each coarse level with the existing Kuramoto coupling.  
4. Computing and saving explicit coarse↔fine index maps for learning.

---

## Next Steps (Prototype)

1. **1D prototype** (space + time):  
   - Two or three scale levels (e.g. 100×500, 50×250, 20×100).  
   - S1 at finest; S2 at each level (or at least at coarsest) with correspondence from S1.  
   - Save S1, S2 per level, K, and correspondence arrays/slices.  

2. **2D extension**: Same idea with `(x, y, t)` and 2D spatial blocks for coarse↔fine.  

3. **Downsampled S1 at coarser levels**: Optional; useful if we want to train a model that sees "S1 at scale 1" and "S2 at scale 1" and learns to align them. Then we need both S1 and S2 at multiple scales, plus the same correspondence maps.

The notebook `multiscale_multimodal_correspondence.ipynb` (in this directory) implements the 1D prototype and validates that the pipeline runs and that correspondence maps are consistent.
