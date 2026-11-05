# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Shared working space for a student team in a Computer Vision course (first semester of a master's program). It serves two purposes at once: **file storage** for the team and a **kanban board** for coordinating work. The end goal is a research paper at the level of a К2–К3 journal (Russian "Белый список" tiers), using SOTA methods.

The current topic is **squat technique assessment from video**. As of 2026-11-05 all 15 REHAB26 squat 3D pose files have been downloaded/audited, and the first nested subject-heldout CPU comparison has run. The primary task has three categories: protocol-correct, knee-inward, shallow squat (388 physical repetitions, 5 people, 1164 synchronized camera observations). Inner-only automatic selection chose logistic regression in all 5 outer folds: mean Macro-F1 .672±.095, false-feedback .211, error recall .742. Guardrails are not met; do not claim readiness for automatic coaching. EC3D remains a historical different five-category task. Checkpoint: 2026-11-11, RTX 2060 6 GB / Colab available, current experiments use CPU. FLEX/Fitness-AQA access is absent.

The source of truth is `docs/project-plan.md`; technical edgecases are in `docs/experiment-protocol.md`, tasks in `docs/roadmap.md`, current results in `reports/rehab26_baseline/README.md`, full pose audit in `reports/rehab26_data_audit.json`. The pre-run protocol/code hashes are saved in `reports/rehab26_baseline/run_config.json`; don't silently revise and present tuned results as untouched test evidence. Historical EC3D findings remain separate. End-to-end raw RGB is not validated: REHAB26 models use prepared MediaPipe `(frames,33,4)` arrays; EC3D models use a different `(frames,25,3)` schema.

`docs/brainstorm.md` is the team's shared fill-in-the-blanks workspace for that decision: a paper-review protocol (how to read a paper in three passes, a card template to fill per paper, a running list of unsolved problems harvested from Limitations sections) plus free-form idea sections. It is a template the team writes into — do not fill it in for them or replace it with conclusions.

`docs/reference-landscape.md` is an archive of earlier metallurgy/video-anomaly topic exploration. Do not treat its old recommendations or numeric journal checklist as current requirements.

## Audience: no prior background

The team has no background in computer vision or in reading research papers, and neither does the rest of their group. Default to step-by-step, plain language, and explain terms on first use. Do not lead with dense surveys of methods, benchmark names, or recommendations the team cannot yet evaluate — they asked explicitly to go slower than that. Scaffolding that helps them reach a conclusion themselves beats handing them the conclusion.

## Current state of the code

Python 3.10/3.11; pinned dependencies in `requirements.txt`. Main pipeline: verified partial ZIP download (`scripts/download_rehab26.py`) → mmap loader (`scripts/rehab26_data.py`) → frozen 35-dimensional global geometry (`scripts/rehab26_features.py`) → nested person-only training (`scripts/rehab26_baseline.py`) → report (`scripts/report_rehab26.py`). Data/models/venv are ignored. CLI prediction without training: `scripts/predict_rehab26.py`. Geometry baseline intentionally ignores visibility and temporalorder; future phase/visibility comparisons must remain separate. Raw zero placeholders exist outside selected intervals, not within the selected 1164 observations.

All new baseline CLI runs use `scripts/experiment_log.py`: unique report/model directories, before-run source/environment snapshot, stdout/stderr log, status including failed/interrupted/prepared, metrics and SHA-256 artifact manifest. Existing nonempty output directories are rejected. Human index: `experiments/README.md`; all-method table: `experiments/comparison.csv`; usage: `docs/experiment-log.md`. Legacy EC3D/REHAB26 reports and weights were copied into separate journal archives before runner changes, preserving original outputs. Initial EC3D training source was not recorded; never claim the import reconstructs missing provenance. REHAB26 reports are generated automatically before sealing a new run; do not rerender or edit a sealed archive. Preserve unsuccessful runs, and do not treat reproducibility reruns as independent evidence.

Since 2026-11-05, training also exports fold-specific model bundles and separate all-data geometry models into ignored `models/ec3d_baseline/`. Bundles include the fitted scaler/classifier, feature recipe version, input contract, training IDs and dependency versions. Fold bundles retain their validation threshold; all-data refits have no calibrated abstention threshold or independent performance estimate. `scripts/predict_squat.py` loads a trusted bundle and predicts from one EC3D `(frames, 25, 3)` NPY sequence without fitting.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_rehab26.py
.venv/bin/python scripts/rehab26_baseline.py --prepare-only
.venv/bin/python scripts/visualize_rehab26.py
.venv/bin/python scripts/rehab26_baseline.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -p test_protocol.py -k test_subjects_never_cross_partitions -v
```

Keep people, repetitions, camera views and their augmented copies together within each split. Fit transformations only on training subjects; choose abstention thresholds only on validation subjects. Report refusals as refusals, never as correct technique. EC3D instruction 10 is undocumented in the paper's table and excluded pending clarification. Literature novelty and end-to-end performance remain unproven; do not fabricate either.

REHAB26 uses semicolon-delimited annotations, correctness 1/5, and a repetition number that resets per orientation. Physical keys are `(video_id, first_frame, last_frame)`; all cameras inherit the same physical key and person split. Do not count three cameras as independent people or map other error types to correct negatives. Main planned metric is person-averaged three-class Macro-F1 at full coverage with explicit false-feedback/error-recall guardrails; thresholds and family selection are inner-only. Phase estimation must use observed/corrupted pose, not hidden clean data. Check PhaseAware 2026 before claiming novelty of phases or masking. Technical choices are the assistant's responsibility; keep user-facing explanations plain and brief.

## Working language

The team works in Russian. Documentation, commit messages, and discussion are in Russian; write new docs in Russian unless asked otherwise. Technical terms and paper/dataset names stay in English.

## Layout note

The git repository root is `cv_semester1/`, a subdirectory of the usual working directory (`CV git/`). The parent is not under version control and holds unrelated course material (`1 лекция.pages`, `2 лекция.pages`). Run git commands from `cv_semester1/` and keep project files inside it.

Remote: `https://github.com/wwwdmtr/cv_semester1.git`, default branch `main`.

## Citations

When referencing literature, do not invent bibliographic data. Paper names, venues, and years in `docs/reference-landscape.md` are navigation aids recorded from memory and are explicitly flagged as needing verification — keep that caveat in place, and verify DOIs and exact details before anything goes into the paper itself.
