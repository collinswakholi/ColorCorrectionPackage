# Local implementation sequence

This is the revised sequence agreed after the architecture review and adversarial
plan review. It replaces the first draft's numbering. Every tranche is implemented
and tested in the local ColorCorrectionPackage clone, then reported for review.
No push or release is part of these tranches.

| Patch | Scope | Status |
| --- | --- | --- |
| 0 | Frozen baseline, fixtures, failure reproductions | Implemented locally |
| 1 | Transactional calibration and truthful outcomes | Implemented locally |
| 2 | Safe persistence, transient LUT repair | Implemented locally |
| 3 | Image/reference/observation contracts | Implemented locally |
| 4 | Explicit color semantics | Implemented locally |
| 5 | Versioned saturation, neutral GC, WB and FFC policies | Implemented locally |
| 6 | Reproducible fitting and grouped, accounted evaluation | Implemented locally |
| 7 | Non-executable versioned calibration bundle | Implemented locally |
| 8 | Complete I/O, batch and reporting parity | Implemented locally |
| 9 | Measured acceleration and distribution qualification | Implemented locally; see [qualification](VALIDATION_9.md) |

Model-based chart detection was separately authorized with patches 0–2, and GPU
selection/CPU fallback with patches 3–4. The Taichi investigation accompanying
patches 5–6 is an experimental benchmark, not a production backend/default change.
Compatibility and new algorithm qualification remain separate acceptance gates.
