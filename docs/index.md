# VGGT-Qwen3 RoomPlan Documentation

This documentation focuses on **Stage 1** of the VGGT-Qwen3 RoomPlan pipeline: multi-view 3D question answering on ScanQA and SQA3D, and its reuse for RoomPlan-style QA. Stage 2 (action JSON prediction) is considered **future work** and is intentionally out of scope for now.

## Contents

- [Stage-1 Quickstart](stage1_quickstart.md)  
  End-to-end instructions to install dependencies, prepare data, train Stage 1, and run QA inference.

- [Model Architecture](model/architecture.md)  
  Technical overview of the VGGT + Perceiver + Qwen3-4B stack, including the token-level visual injection mechanism and training objective.

- [Developer / Maintainer Notes](dev/repo_structure.md)  
  How the repository is organized, how entry points are wired, and where to add new components.

- [Debug History](dev/debug_history.md)  
  A chronological record of key engineering/debugging episodes for Stage 1 (e.g., `<image>` placeholder handling, injection span issues). Useful for maintainers, not required for first-time users.

If you are new to the repo, start with **Stage-1 Quickstart**, then refer to **Model Architecture** for deeper details.

