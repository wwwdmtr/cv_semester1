# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Shared working space for a student team in a Computer Vision course (first semester of a master's program). It serves two purposes at once: **file storage** for the team and a **kanban board** for coordinating work. The end goal is a research paper at the level of a К2–К3 journal (Russian "Белый список" tiers), using SOTA methods.

The current topic, fixed on 2026-11-04, is **squat technique assessment from video**. The first experiment uses the accessible EC3D skeleton dataset. The deadline is one week (checkpoint 2026-11-11); resources are RTX 2060 6 GB or Colab. Access to FLEX/MyoMechanix and Fitness-AQA has not been obtained.

The source of truth is `docs/project-plan.md`; tasks are in `docs/roadmap.md`, verified literature notes in `docs/research-review-2026-11-04.md`, and executed baseline findings in `reports/ec3d_baseline/README.md`. The end-to-end RGB pipeline is not yet validated: the implemented classifier currently takes prepared EC3D 3D skeletons.

`docs/brainstorm.md` is the team's shared fill-in-the-blanks workspace for that decision: a paper-review protocol (how to read a paper in three passes, a card template to fill per paper, a running list of unsolved problems harvested from Limitations sections) plus free-form idea sections. It is a template the team writes into — do not fill it in for them or replace it with conclusions.

`docs/reference-landscape.md` is an archive of earlier metallurgy/video-anomaly topic exploration. Do not treat its old recommendations or numeric journal checklist as current requirements.

## Audience: no prior background

The team has no background in computer vision or in reading research papers, and neither does the rest of their group. Default to step-by-step, plain language, and explain terms on first use. Do not lead with dense surveys of methods, benchmark names, or recommendations the team cannot yet evaluate — they asked explicitly to go slower than that. Scaffolding that helps them reach a conclusion themselves beats handing them the conclusion.

## Current state of the code

Python 3.10/3.11; dependencies are pinned in `requirements.txt`. The pipeline is: verified download (`scripts/download_ec3d.py`) → grouped sequences → normalization/features → subject-separated classifiers and JSON/CSV/PNG reports (`scripts/ec3d_baseline.py`). Raw data and the virtual environment are ignored by Git.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_ec3d.py
.venv/bin/python scripts/ec3d_baseline.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -p test_protocol.py -k test_subjects_never_cross_partitions -v
```

Keep people, repetitions, camera views and their augmented copies together within each split. Fit transformations only on training subjects; choose abstention thresholds only on validation subjects. Report refusals as refusals, never as correct technique. EC3D instruction 10 is undocumented in the paper's table and excluded pending clarification. Literature novelty and end-to-end performance remain unproven; do not fabricate either.

## Working language

The team works in Russian. Documentation, commit messages, and discussion are in Russian; write new docs in Russian unless asked otherwise. Technical terms and paper/dataset names stay in English.

## Layout note

The git repository root is `cv_semester1/`, a subdirectory of the usual working directory (`CV git/`). The parent is not under version control and holds unrelated course material (`1 лекция.pages`, `2 лекция.pages`). Run git commands from `cv_semester1/` and keep project files inside it.

Remote: `https://github.com/wwwdmtr/cv_semester1.git`, default branch `main`.

## Citations

When referencing literature, do not invent bibliographic data. Paper names, venues, and years in `docs/reference-landscape.md` are navigation aids recorded from memory and are explicitly flagged as needing verification — keep that caveat in place, and verify DOIs and exact details before anything goes into the paper itself.
