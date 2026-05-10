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
