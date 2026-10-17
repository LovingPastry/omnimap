# Agent Rules

## Repo Purpose
OmniMap builds an online map by coupling three modalities: optical appearance (3DGS), geometry (voxel/TSDF-like structure), and semantics (open-vocabulary understanding).

## Core Principle
Keep the three streams synchronized frame-by-frame so mapping, decision, and control can run in closed loop.

## Current ROS Focus
Primary focus is `info_flow/`, specifically the three loops:
- tracking
- planning
- servo

## Build/Test Environment Note
- For quick compile/sanity checks, use conda env `InfoFlow`.
- Ignore `open3d`/`torch` import or package errors during these checks; exec-side and compute-side envs differ.

## Mandatory Git Step After Code Changes
After any code edit in this workspace, run:

```bash
git add . && git commit -m "<commit>" && git push
```
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 80
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/fuyx/lanzc/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'	' '$1=="repo_root"{print $2; exit}' "/home/fuyx/lanzc/omnimap/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/fuyx/lanzc/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/home/fuyx/lanzc/omnimap" --reconcile`
For copied Codex installs, use:
`bash /home/fuyx/lanzc/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/home/fuyx/lanzc/omnimap"`
<!-- ARIS-CODEX:END -->