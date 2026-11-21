# Open-source capacitance extractors

Date: 2026-11-15

> Compiled from the public pages of each project in November 2026; not exhaustive.
> Licenses and details not stated on those pages are marked "Unknown". Corrections are welcome.

## 1. Tools

| Tool | Method | Language | License | Multilayer dielectrics | Parallelism | Status / notes |
|---|---|---|---|---|---|---|
| **RWCap v4** (Tsinghua numbda) | FRW 3D | Unknown | Unknown | Per the papers, supports conformal and non-stratified dielectrics | Unknown | 2023 release. Apparently distributed only as binaries in a tar.gz; availability of source unconfirmed. Input is `.cap3d` / qbem |
| **deepRWCap** (THU-numbda) | FRW + CNN | C++ (LibTorch/TensorRT) + Python | MIT | ○ (takes permittivity distribution as input) | CPU multicore + GPU required | Research code for AAAI 2026. Requires CUDA 12.6+/TensorRT |
| **CNNCap** | CNN prediction (2D/3D patterns) | Python (PyTorch) | Not stated | — | — | Trained models, not a solver |
| **FasterCap** (FastFieldSolvers) | BEM (FMM family), 2D/3D | C++ | LGPL-2.1 | ○ | OpenMP (inferred) | Has GUI and CLI. Repository is rarely updated |
| **FastCap2** (MIT) | BEM + FMM | C | Custom (old MIT) | ○ (discretizes dielectric interfaces) | × | 1990s code. Still used as a reference solution |
| **CAPLET** (MIT) | BEM + instantiable basis | C++ / Python2 | LGPL/GPL-3 | **× (uniform dielectric, fixed ε=1)** | MPI / OpenMP | No updates since 2013. Has GDS input |
| **OpenRCX** (OpenROAD) | Pattern-calibrated 2.5D (rules file) | C++ | BSD-3 | Absorbed during calibration | ○ | Generating rules requires output from an external field solver |
| **Magic ext** | Analytical 2.5D | C | BSD-style | Simplified | × | Standard for OSS PDKs such as sky130 / gf180 |
| **KLayout-PEX** (JKU) | FasterCap integration / Magic wrapper / 2.5D (in development) | Python | GPL-3.0 | Depends on tool | Depends on tool | The authors themselves state "early stage, not intended for production use" |
| **SPACE** (TU Delft) | BEM 3D + 2D mode | C | Custom | ○ | × | Old academic tool |
| **Palace** (AWS) | FEM (MFEM) electrostatics | C++ | Apache-2.0 | ○ | MPI / GPU | General-purpose EM solver for quantum devices. Can extract capacitance matrices but is not aimed at VLSI full-chip |
| Elmer / gdsfactory gplugins | FEM wrapper | — | — | ○ | — | General-purpose FEM. Not aimed at VLSI extraction |

Legend: ◎ excellent, ○ good, △ fair, × poor.

## 2. Where faradice fits

faradice is an FRW extractor with public source under a permissive license (MIT OR
Apache-2.0). Within the search above:

- Of the FRW tools, RWCap is distributed as binaries without a stated license, and
  deepRWCap is research code that requires a GPU and TensorRT.
- The tools used with open PDKs today are analytical or pattern-calibrated 2.5D models
  (Magic, OpenRCX) and boundary-element solvers (FasterCap, and FasterCap through
  KLayout-PEX).
- We found no capacitance extractor written in Rust (searches of GitHub and crates.io).

The FRW method itself is well established, mainly through the RWCap line of papers; see
the [survey](01_capacitance_extraction_survey.md) §3.3 and the design documents for the
published methods faradice builds on.

## References

- RWCap v4: https://numbda.cs.tsinghua.edu.cn/download/RWCap_v4_en.html
- deepRWCap: https://github.com/THU-numbda/deepRWCap
- CNNCap: https://github.com/ydc123/CNNCap
- CAPLET: https://github.com/yuchsiao/caplet
- FasterCap: https://github.com/ediloren/FasterCap
- OpenRCX: https://openroad.readthedocs.io/en/latest/main/src/rcx/README.html
- KLayout-PEX: https://iic-jku.github.io/klayout-pex-website/doc/doc.html , https://pypi.org/project/klayout-pex/
- Palace: https://awslabs.github.io/palace/stable/examples/spheres/ , https://aws.amazon.com/blogs/quantum-computing/aws-releases-open-source-software-palace-for-cloud-based-electromagnetics-simulations-of-quantum-computing-hardware/
- gplugins Palace: https://gdsfactory.github.io/gplugins/_modules/gplugins/palace/get_capacitance.html
- SPACE (TU Delft): https://space.tudelft.nl/
