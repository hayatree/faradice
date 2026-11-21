# Capacitance extraction methods: a survey

Date: 2026-11-15

> This survey summarizes published papers and public product descriptions. The internal
> algorithms of commercial extractors are not public, and nothing here describes them. All
> numbers are as reported by the respective papers or vendors.

---

## 1. Problem Statement

- Apply 1 V to conductor i and 0 V to all other conductors, solve the Laplace equation `∇⋅(ε∇φ) = 0` in the
  dielectric, and obtain the capacitance matrix (Maxwell capacitance matrix) from the charge on each conductor
  surface, `Q_j = ∮ ε ∂φ/∂n dS`.
  - Diagonal: total (self) capacitance; off-diagonal: coupling capacitance.
- What makes advanced nodes hard
  - Multilayer, high-contrast dielectrics (low-k, etch-stop layers, **conformal dielectric**, air gap)
  - FinFET / GAA MEOL and near-device 3D geometry (gate–contact, gate–fin capacitance)
  - Process effects: width/spacing-dependent etch bias, density-dependent thickness variation (CMP),
    trapezoidal cross-sections, multi-patterning
  - Combining full-chip scale (hundreds of millions to billions of polygons) with sign-off accuracy
    (~1% total capacitance, ~5% for dominant coupling capacitances)

## 2. Industrial practice

Commercial sign-off extractors combine precomputed tables for 2.5D extraction of
cross-section patterns with 3D field solvers for critical nets, analog and custom blocks,
and regions near devices. A volume-discretization solver serves as the accuracy
reference. Open-source flows use analytical or pattern-calibrated 2.5D models (Magic,
OpenRCX) or general boundary-element solvers (FastCap, FasterCap).

## 3. Field Solver Methods

### 3.1 Volume Discretization: FDM / FEM (Raphael, etc.)

- Handles arbitrary dielectrics and geometry directly; serves as the accuracy reference.
- Unknowns scale with volume; slow and memory-hungry at large scale. Not suited to full-chip.
- Use in faradice: reference values for small benchmark cases, table generation, and regression tests.

### 3.2 Boundary Element Method (BEM)

- Discretizes only conductor/dielectric interfaces. The matrix is dense → acceleration is essential.
  - **FastCap** (Nabors & White 1991): GCR iteration + FMM. O(N) to O(N log N).
  - **pFFT** (pre-corrected FFT), **H-matrix / IES3 / wavelet** (Tausch & White), etc.
  - **FasterCap**: Improvements on the FastCap family + hierarchical methods (OSS).
  - **Instantiable basis functions / CAPLET** (Hsiao & Daniel, MIT): Uses physical charge-distribution shapes
    as basis functions, reducing unknowns by ~30x. Sub-second / within 5% for tens of wires.
  - **FMMGpu**: FMM capacitance extraction on GPU.
- Challenges: With multilayer dielectrics, either (a) dielectric interfaces must also be discretized → more
  unknowns, or (b) a multilayer Green's function is needed (requires series expansion / tabulation, and is
  unsuitable for non-planar layers such as conformal dielectrics). Struggles with FinFET high-contrast
  conformal dielectrics.
- Strong for obtaining the full capacitance matrix (all nets) at once and for precise small-to-medium-scale
  analysis.

### 3.3 Floating random walk (FRW)

- Principle: The potential at a point equals the Green's-function-weighted average of the potential over a
  surrounding closed surface. Starting from a point on the Gaussian surface enclosing conductor i, place "the
  largest cube that does not touch any conductor (transition cube)", jump stochastically to its surface, and
  repeat … when the walk reaches conductor j, it contributes to C_ij. Estimated by Monte Carlo.
- Representative work: **RWCap** (Yu et al., Tsinghua University, IEEE TCAD 2013 onward). Huang & Yu's
  **FRW-AGF**, **Microwalk**, **DeepRWCap** (2025, predicts transition quantities with a CNN; average error
  1.24% vs. Raphael on FinFETs with high-contrast conformal dielectrics), and variance reduction by
  **generalized antithetic sampling** (up to 30% fewer walks, 2025).
- Strengths (very good fit for faradice's goals)
  - **Mesh-free**; memory is only the geometry (spatial index) → light even at full-chip scale.
  - **Nets are computed independently**, and walks are mutually independent → almost perfectly parallel
    (multicore / GPU / distributed).
  - **Statistically controlled error** (stop once the target accuracy is reached) → explicit accuracy/speed
    trade-off.
  - Multilayer dielectrics are handled by "precomputing and tabulating Green's functions for multi-dielectric
    transition cubes" (RWCap's approach).
  - CapBench (2026) uses RWCap as its labeling engine and reports it as more than two orders of magnitude
    faster than Raphael, with total capacitance error < 1% and coupling ~2%.
- Weaknesses and implementation essentials
  - Small coupling capacitances tend to have large relative error (variance reduction and importance sampling
    are important).
  - The per-step "nearest-conductor distance query" is the dominant cost → **spatial index (octree / grid /
    BVH) performance is everything**.
  - Steps become small near dielectric interfaces → multi-dielectric cube tables, interface-specific
    transitions.
  - Results have random fluctuation (can be made deterministic by fixing the seed).

### 3.4 ML-Based

| Method | Year | Summary | Reported results |
|---|---|---|---|
| CNN-Cap | 2021 ICCAD / TCAD | Rasterizes 2D cross-section patterns into grid images and predicts with a ResNet-style CNN (replaces the tables in 2.5D extraction) | Total capacitance error <1.3%, 99.5% of coupling capacitances within 10% error, 4000x vs. 2D solver (GPU) |
| GNN-Cap | 2024 TCAD | Converts chip-scale interconnect into a graph and predicts total/coupling capacitance with a GNN | 11–13x faster than StarRC (reported) |
| DeepRWCap | 2025 | Hybrid that predicts FRW transition quantities with a CNN | Average 1.24% vs. Raphael |
| AttentionCap | 2026 | Infers the capacitance matrix within a window in one pass with a Transformer | self 0.67% / coupling 3.99% (unverified) |
| CapBench | 2026 | Multi-PDK dataset + 10 ML baselines | — |
| Tsai et al. | 2025 ICCAD | ML estimation of standard-cell parasitics | — |

- Assessment: Fast, but requires training per PDK and offers no accuracy guarantee for out-of-distribution
  geometry → unsuitable as the main sign-off engine. Realistic uses are **"an accelerator component inside a
  physical solver" (DeepRWCap style) or "a replacement for pattern tables" (CNN-Cap style)**.

## 4. Method Comparison

| Aspect | FDM/FEM | BEM (FMM/pFFT) | FRW | 2.5D pattern matching | ML |
|---|---|---|---|---|---|
| Accuracy | ◎ (reference) | ◎ | ◎ (statistical error control) | ○ (pattern-dependent) | △ to ○ |
| Speed (large scale) | × | △ | ○ | ◎ | ◎ |
| Memory | × | △ | ◎ | ◎ | ○ |
| Multilayer / conformal dielectrics | ◎ | △ | ○ (tabulation) | ○ (absorbed by precomputation) | △ |
| Parallelism | △ | ○ | ◎ (embarrassingly parallel) | ◎ | ◎ (GPU) |
| Precomputation | None | None | Dielectric cube tables | Large, per process | Large training effort |
| Implementation difficulty | Medium | High | Medium | Medium (needs a 3D solver for table generation) | Medium + data |

Legend: ◎ excellent, ○ good, △ fair, × poor.

## References and Sources

- RWCap: W. Yu, H. Zhuang, C. Zhang, G. Hu, Z. Liu, "RWCap: A Floating Random Walk Solver for 3-D Capacitance Extraction of Very-Large-Scale Integration Interconnects", IEEE TCAD 32(3), 2013 — https://doi.org/10.1109/TCAD.2012.2224346
- DeepRWCap (2025): https://arxiv.org/pdf/2511.06831
- Generalized antithetic sampling for FRW (2025): https://arxiv.org/abs/2504.20586v1
- CapBench (2026): https://arxiv.org/pdf/2604.11202
- CNN-Cap: https://arxiv.org/pdf/2107.06511
- GNN-Cap: L. Liu, F. Yang, L. Shang, X. Zeng, "GNN-Cap: Chip-Scale Interconnect Capacitance Extraction Using Graph Neural Network", IEEE TCAD 43, 2024 — https://doi.org/10.1109/TCAD.2023.3331942
- AttentionCap: J. Huang et al., "AttentionCap: Transformer Based Capacitance Matrix Learning Toward Full-Chip Extraction", DAC 2026 — https://arxiv.org/abs/2606.08161
- Tsai et al., ICCAD 2025: https://gtcad.gatech.edu/www/papers/Tsai-ICCAD25.pdf
- FastCap: https://www.layouteditor.org/info/fastcap/tcad.pdf
- CAPLET / instantiable basis functions: https://mtlsites.mit.edu/annual_reports/2013/caplet-field-solver-accurate-real-time-capacitance-extraction-toolkit-using-instantiable-basis-functions
- Parallel BEM (MIT): https://mtlsites.mit.edu/annual_reports/2011/a-highly-scalable-parallel-boundary-element-method-for-capacitance-extraction-2
- Multilayer Green's function BEM (ASP-DAC 2006): https://www.aspdac.com/aspdac2006/archives/pdf/7B-3.pdf
- StarRC Custom Rapid3D whitepaper: https://www.synopsys.com/implementation-and-signoff/resources/whitepapers/starrc-custom-rapid3d.html
- StarRC Custom datasheet: https://www.synopsys.com/content/dam/synopsys/implementation&signoff/datasheets/starrc-custom-ds.pdf
- Rapid3D 20x announcement article: https://www.electronicspecifier.com/?p=112220
- Overview of parasitic extraction (EE Journal 2010): https://www.eejournal.com/article/20100713-extraction/
- Calibre xACT 3D fact sheet: https://resources.sw.siemens.com/en-US/fact-sheet-calibre-xact-3d

