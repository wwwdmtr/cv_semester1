# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Shared working space for a student team in a Computer Vision course (first semester of a master's program). It serves two purposes at once: **file storage** for the team and a **kanban board** for coordinating work. The end goal is a research paper at the level of a К2–К3 journal (Russian "Белый список" tiers), using SOTA methods.

As of now the team is still **choosing a research topic** between two candidates:
1. Defect detection in metallurgy (surface defect / industrial anomaly detection).
2. Video analysis for antisocial behaviour (video anomaly / aggression detection).

`docs/brainstorm.md` is the team's shared fill-in-the-blanks workspace for that decision: a paper-review protocol (how to read a paper in three passes, a card template to fill per paper, a running list of unsolved problems harvested from Limitations sections) plus free-form idea sections. It is a template the team writes into — do not fill it in for them or replace it with conclusions.

`docs/reference-landscape.md` is a dense analytical companion (datasets, SOTA methods, novelty angles, a recommendation, journal checklist) written before the team's experience level was clear. It is deliberately parked as optional later reading.

## Audience: no prior background

The team has no background in computer vision or in reading research papers, and neither does the rest of their group. Default to step-by-step, plain language, and explain terms on first use. Do not lead with dense surveys of methods, benchmark names, or recommendations the team cannot yet evaluate — they asked explicitly to go slower than that. Scaffolding that helps them reach a conclusion themselves beats handing them the conclusion.

## Current state of the code

No source code, build system, dependencies, or tests yet — the repository holds documentation only. There is nothing to build, lint, or test. Once a toolchain exists, document the real commands here (including how to run a single test) and add an architecture overview.

## Working language

The team works in Russian. Documentation, commit messages, and discussion are in Russian; write new docs in Russian unless asked otherwise. Technical terms and paper/dataset names stay in English.

## Layout note

The git repository root is `cv_semester1/`, a subdirectory of the usual working directory (`CV git/`). The parent is not under version control and holds unrelated course material (`1 лекция.pages`, `2 лекция.pages`). Run git commands from `cv_semester1/` and keep project files inside it.

Remote: `https://github.com/wwwdmtr/cv_semester1.git`, default branch `main`.

## Citations

When referencing literature, do not invent bibliographic data. Paper names, venues, and years in `docs/reference-landscape.md` are navigation aids recorded from memory and are explicitly flagged as needing verification — keep that caveat in place, and verify DOIs and exact details before anything goes into the paper itself.
