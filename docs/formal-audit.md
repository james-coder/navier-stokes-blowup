# Independent upstream formal audit

This audit targets [OpenAI/NavierStokesAndEuler at `8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538`](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538). The numerical Python code is not part of that Lean repository and is not certified by building or checking it.

**Both audits passed on 2026-09-10.** The full repository build completed 11,251 jobs with exit status 0. The separate Navier–Stokes Comparator procedure matched both selected statements and allowed axioms, and both nanoda and Lean's default kernel accepted the exported solution. Comparator exited with status 0 and printed `Your solution is okay!`.

[Download the complete build, tool-setup and all Comparator attempt logs](formal-audit-logs.zip) and [read the machine-readable versions, hashes and results](formal-audit.json). The latter records the 1,071,397,994-byte solution export's SHA-256; that large temporary export is not committed to this repository. Both source checkouts remained clean. The successful Comparator invocation took 12 minutes 53.493 seconds after the earlier sandbox compilation; this is not a clean-install benchmark.

The full build includes Euler, but **the separate Euler Comparator challenge was not run**. This audit's independent comparison scope is the two Navier–Stokes declarations below.

## Source and environment

| Component | Pinned version or commit |
|---|---|
| Upstream project | `8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538` |
| Lean / mathlib | `v4.34.0-rc2` |
| Comparator | `19e111e2141cf333c7daff0f64c5f24acc91dd2e` |
| lean4export | `cacf989bd75f608700820f6afc595f32e7a99a4d` |
| landrun | `811cfff51ceaf3d9843708aa6d22e9b84ccac8b4` |
| nanoda | `4c544ed4099c8227f07d5de77ad1e69fb0740a27` |
| Elan installer | `v4.2.4` |
| Go compiler | `go1.27.1 linux/amd64` |
| Rust compiler | `rustc 1.93.0 (254b59607 2026-01-19)` |
| Linux kernel | `6.6.87.2-microsoft-standard-WSL2` |

Toolchains, source checkouts, dependency caches, builds and exports are isolated under a task directory on attached ext4 storage. The build checkout and Comparator checkout are separate. The latter starts from fresh pinned source and trusted mathlib cache; the solution is compiled there only by Comparator's sandboxed procedure. These are separate directories on the same non-root workstation account, not independent hardware or virtual-machine trust domains.

The pinned [Navier–Stokes Comparator configuration](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/ComparatorChallenges/NavierStokes.json) compares these exact declarations:

- `NavierStokes.Comparator.navier_stokes_breakdown_R3`
- `NavierStokes.Comparator.navier_stokes_breakdown_periodic`

Its permitted axioms are `propext`, `Quot.sound`, and `Classical.choice`; `enable_nanoda` is true. The [submission module](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ComparatorSolution.lean) exposes the whole-space and periodic nonexistence statements for every positive viscosity and prints their axiom dependencies. The challenge modules deliberately contain `sorry` placeholders: warnings from those reference statements must be distinguished from unproved solution declarations.

## Reproduction

Install [Elan v4.2.4](https://github.com/leanprover/elan/releases/tag/v4.2.4) without modifying global shell configuration, and point `ELAN_HOME` at storage suitable for the toolchain. The pinned `lean-toolchain` selects Lean 4.34.0-rc2. A fresh checkout runs the upstream commands:

```bash
git clone --no-checkout https://github.com/openai/NavierStokesAndEuler.git
cd NavierStokesAndEuler
git checkout --detach 8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538
lake exe cache get
lake build
```

The default build covers `NavierStokes`, `Euler`, and `ComparatorChallenges`. A successful compile is a kernel-checked build under Lean's trust assumptions; it does not by itself independently compare the formal statement to a challenge or replay it in another kernel.

For Comparator, build the pinned [landrun](https://github.com/Zouuup/landrun/tree/811cfff51ceaf3d9843708aa6d22e9b84ccac8b4) with Go (`go build -o /path/to/tools/landrun ./cmd/landrun`), and the pinned [nanoda](https://github.com/ammkrn/nanoda_lib/tree/4c544ed4099c8227f07d5de77ad1e69fb0740a27) with `cargo build --release`. In the pinned Comparator package, `lake build lean4export comparator` builds the other tools. **lean4export is placed under its own dependency package's `.lake/build/bin`, not Comparator's `.lake/build/bin`.** Put that binary, `comparator`, `landrun`, and `nanoda_bin` together on a tools path or use their absolute paths.

Follow the [pinned Comparator trust and sandbox instructions](https://github.com/leanprover/comparator/blob/19e111e2141cf333c7daff0f64c5f24acc91dd2e/README.md). On this kernel, the prescribed `RestrictAddressFamilies=~AF_UNIX` systemd restriction is required in addition to landrun. Its operation was checked before the audit. Do not use a fake landrun wrapper or disable the external kernel to turn a failure into a pass.

The actual checking command has this form, in a **fresh** source checkout:

```bash
systemd-run --user --wait --pipe \
  --property=RestrictAddressFamilies=~AF_UNIX \
  --working-directory="$PWD" \
  -E PATH="$PATH" -E ELAN_HOME="$ELAN_HOME" -E TMPDIR="$TMPDIR" \
  -E COMPARATOR_LANDRUN=/path/to/tools/landrun \
  -E COMPARATOR_LEAN4EXPORT=/path/to/tools/lean4export \
  -E COMPARATOR_NANODA=/path/to/tools/nanoda_bin \
  -- lake env /path/to/tools/comparator ComparatorChallenges/NavierStokes.json
```

Set `TMPDIR` to a directory on bulk storage before the manual command; Comparator writes its external-kernel export there. Use a fresh solution checkout and retain the combined stdout/stderr plus process exit status for each command. The first local Comparator attempt failed before solution compilation because the lean4export binary path was wrong. The second used the correct path and compiled the solution inside the sandbox, then was stopped before export to set `TMPDIR` explicitly. The third retained those sandbox-built modules and the same pinned source. All three attempt logs are included in the downloadable archive; the stopped second run is not counted as a passing audit.

## What the checks can establish

Under Comparator's documented trust assumptions, success establishes that the selected solution declarations match the challenge statements, use only the permitted axioms, and pass the configured kernel checks. Trust still includes the chosen challenge definitions, the sandbox and operating system, hardware, export/checking implementations, and at least one correct checking kernel. Comparing the statements' mathematical intent to the paper is a further review task.

No outcome here certifies finite-precision Python translation, numerical truncation, a complete executable blowup field, or a fast general-purpose Navier–Stokes algorithm. See the [construction map](construction-map.md) for what remains to be translated and the [numerical validation record](validation.md) for the evidence actually applying to this library.
