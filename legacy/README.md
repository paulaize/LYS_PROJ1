# Legacy context code

`legacy/context_code/` contains older local scripts that informed this project.
They are useful reference material, but they are **not production pipeline code**.

Known issues in these scripts may include:

- hardcoded local paths,
- exploratory atlas/brainreg code that belongs to later milestones,
- TIFF/ND image workflows that are not the v1 `.vsi` QuPath path,
- older variable names that imply FITC = LYS241 concentration.

Do not import these files from `src/`. Extract ideas only through small, reviewed refactors that follow `AGENTS.md`.
