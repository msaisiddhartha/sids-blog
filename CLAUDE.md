# CLAUDE.md

Personal tech blog built with **Hugo** (static site generator), themed with **PaperMod**,
deployed to **GitHub Pages** at https://msaisiddhartha.github.io/sids-blog/.

## Stack & key facts
- **Hugo** extended (local: v0.152.2). Config: `hugo.yaml` (YAML, not TOML).
- **Theme**: PaperMod, pulled as a git submodule at `themes/PaperMod` (see `.gitmodules`).
  Do NOT edit files under `themes/PaperMod` — override via `layouts/` instead.
- **Math**: MathJax, enabled per-post with `math: true` in front matter. Loaded by
  `layouts/partials/extend_head.html`. IMPORTANT: `hugo.yaml` enables the Goldmark
  **passthrough** extension for `$...$` / `$$...$$` — without it, Hugo's markdown eats
  underscores and `\;` inside equations and the math renders wrong (silently, no error).
- **Deploy**: push to `main` → `.github/workflows/deploy.yml` builds with `hugo --minify`
  and publishes `./public` to GitHub Pages. No manual deploy needed.

## Layout
- `content/posts/` — blog posts (Markdown + YAML front matter). Currently one post:
  `parametric-optimization-neural-networks.md`.
- `static/images/` — images referenced by posts. Because `baseURL` has the `/sids-blog/`
  subpath, reference images with the full subpath (this was a past bug — see git log).
- `layouts/partials/` — site-level template overrides (only `extend_head.html` so far).
- `archetypes/default.md` — front-matter template for `hugo new`.
- `assets/`, `data/`, `i18n/` — present but empty/unused.
- `public/`, `resources/`, `.hugo_build.lock` — Hugo build output, git-ignored.
- `.venv/` — Python virtualenv, git-ignored.

## Post front matter (typical)
```yaml
---
title: "..."
date: 2025-12-27T14:15:00-05:00
draft: false
tags: ["Machine Learning", "Optimization"]
categories: ["Tech"]
math: true          # only if the post uses LaTeX
ShowToc: true
TocOpen: true
---
```

## Python scripts (repo root)
Standalone PyTorch demos that generated the content/figures for the surrogate-optimization
post — NOT part of the Hugo build:
- `final_demo.py` — trains a neural-net surrogate of a black-box function, then optimizes
  the input via gradient descent (treating input x as the trainable parameter).
- `seed_test.py` — runs the same experiment across seeds to check robustness.

## Figure pipeline for the CAE 3D-representations post (`scripts/`)
- `scripts/figlib.py` — shared paths, matplotlib style, mesh loading + cheap vertex-cluster
  decimation, SDF slice, etc.
- `scripts/make_figures.py` — one function per figure; writes `static/images/cae_*.png`.
  Run: `.venv/bin/python3 scripts/make_figures.py [names...]` (no args = all 7 figures).
- `scripts/make_topology_gif.py` — animated 3D level-set sweep of a bracket's SDF
  (voxelize → `scipy` distance transform → `skimage` marching_cubes at swept iso-levels →
  PIL GIF). Writes `static/images/cae_topology_sweep.gif` (~2.8 MB). Fast (~30s) because it
  uses the distance transform, NOT slow trimesh proximity. Run: `.venv/bin/python3
  scripts/make_topology_gif.py [bracket_id]` (default 148).
- Reads the **SimJEB** bracket dataset from `~/datasets/simjeb/` (outside the repo, ~1.8 GB,
  do NOT commit). Bracket #148 is the through-line; `all_bracket_metadata.csv` drives the
  surrogate scatter. The SDF figure is slow (~3 min, uses `rtree` + trimesh proximity).
- Only the rendered PNGs go in `static/images/` and get committed — never the raw meshes.
- `static/images/ref_*.png` are NOT script-generated: they're figures pulled from the cited
  papers (PointNet, MeshGraphNets, DeepSDF, 3D-GAN, DeepCAD) via ar5iv, embedded with
  "(Image source: …)" attribution captions. Third-party copyrighted — fair-use/educational;
  confirm before relying on them in anything beyond this personal blog.

## Python venv note
`.venv/` runs **Python 3.13.7** (Homebrew, `/usr/local/bin/python3.13`), recreated via
`python3.13 -m venv .venv`. Packages: numpy, scipy, matplotlib, trimesh, scikit-image,
pandas, networkx, rtree, fast-simplification. Always invoke as `.venv/bin/python3`.
(History: the original venv was Python 3.9 and broke when the folder was renamed
"My Files" → "Mac Files"; upgraded to 3.13 so `fast-simplification` works.)

## Common commands
```bash
hugo server -D              # local preview (drafts included) at localhost:1313
hugo new posts/my-post.md   # scaffold a new post
hugo --minify               # production build into ./public (CI does this)
git submodule update --init --recursive   # after fresh clone, fetch theme
```

## Notes
- Images must include the `/sids-blog/` subpath or they 404 on the deployed site.
- New posts default to `draft: true` (from the archetype); set `draft: false` to publish.
```
