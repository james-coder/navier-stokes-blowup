# Issue backlog

The [GitHub backlog index](https://github.com/james-coder/navier-stokes-blowup/issues/1) tracks unfinished implementation, validation gaps, and future ideas.

The 2026-09-10 audit opened **101 actionable issues**, **9 area trackers**, and **one index** (111 issues total), based on source commit `28ed6d444c91`. This is an audit snapshot; GitHub issue states are the current source of truth.

| Area | Tracker | Actionable issues |
| --- | --- | --- |
| Complete the executable blowup construction | [#2](https://github.com/james-coder/navier-stokes-blowup/issues/2) | 15 |
| Establish independent references and mathematical validation | [#3](https://github.com/james-coder/navier-stokes-blowup/issues/3) | 13 |
| Build the full Singularity Observatory solver benchmark | [#4](https://github.com/james-coder/navier-stokes-blowup/issues/4) | 9 |
| Extend and qualify the reusable numerical solver | [#5](https://github.com/james-coder/navier-stokes-blowup/issues/5) | 7 |
| Validate accelerators and investigate performance backends | [#6](https://github.com/james-coder/navier-stokes-blowup/issues/6) | 13 |
| Make the Observatory and library broadly usable | [#7](https://github.com/james-coder/navier-stokes-blowup/issues/7) | 11 |
| Explore geometry, multiphysics, and Blender workflows | [#8](https://github.com/james-coder/navier-stokes-blowup/issues/8) | 16 |
| Explore validated differentiable inverse-design applications | [#9](https://github.com/james-coder/navier-stokes-blowup/issues/9) | 10 |
| Close documentation, compatibility, and release-validation gaps | [#10](https://github.com/james-coder/navier-stokes-blowup/issues/10) | 7 |

The requested **ScreenCLI installation/demo/process recording** is [#11](https://github.com/james-coder/navier-stokes-blowup/issues/11).

## How the work is organized

The individual tasks comprise 36 implementation tasks, 31 validation tasks, 30 optional ideas, and 4 documentation/demonstration tasks. Each has a current-gap description, at least three completion criteria, a pinned source reference, an area tracker, and prerequisite links where relevant.

- `priority: core`: the complete construction, independent reference evidence, and the solver-tracking flagship.
- `priority: next`: library qualification, usability, compatibility, and documentation.
- `priority: future` / `idea`: optional extensions to investigate; these are not release commitments or capability claims.
- `validation`: the issue identifies the missing evidence and preserves the scope of narrower checks that already pass.

The audit covers the README status matrix, all roadmap stages, research and platform assessments, API/numerical limitations, the validation record, the imported conversation, and the additional ScreenCLI request. Related mentions were consolidated into distinct deliverables rather than duplicated.

Existing CPU/CUDA checks, the public repository and transcript, CPU CI, package/documentation builds, Trusted Publishing, and the successful 0.1.0 release remain completed work. New issues cover additional scope and validation; they do not reset those accomplishments.

Update capability claims with evidence as issues are resolved. A finite numerical check is not proof of a limiting mathematical claim, and an optional application idea is not an implemented feature.
