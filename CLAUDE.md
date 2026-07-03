# CLAUDE.md

Personal tech blog built with **Hugo** (static site generator), themed with **PaperMod**,
deployed to **GitHub Pages** at https://msaisiddhartha.github.io/sids-blog/.

## Stack & key facts
- **Hugo** extended (local: v0.152.2). Config: `hugo.yaml` (YAML, not TOML).
- **Theme**: PaperMod, pulled as a git submodule at `themes/PaperMod` (see `.gitmodules`).
  Do NOT edit files under `themes/PaperMod` — override via `layouts/` instead.
- **Math**: MathJax, enabled per-post with `math: true` in front matter. Loaded by
  `layouts/partials/extend_head.html`.
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
