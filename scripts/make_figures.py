"""Generate every figure for the "3D Representations for CAE" post.

Usage:
    .venv/bin/python3 scripts/make_figures.py            # all figures
    .venv/bin/python3 scripts/make_figures.py spine sdf  # a subset

Reads the SimJEB bracket #148 mesh + the 381-row metadata from
~/datasets/simjeb and writes PNGs into static/images/. Deterministic
(fixed RNG seed) so re-runs are reproducible.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

import figlib as F

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- spine
def fig_spine():
    """Hero figure: bracket #148 shown as the five representations."""
    m = F.load_bracket()
    disp = F.decimate(m, grid=70)

    fig = plt.figure(figsize=(19, 4.6))
    titles = ["grids\n(voxels)", "sets\n(point cloud)", "graphs\n(mesh)",
              "fields\n(SDF slice)", "programs\n(CAD B-rep)"]

    # 1. voxels
    ax = fig.add_subplot(1, 5, 1, projection="3d")
    vg = m.voxelized(pitch=m.extents.max() / 22).fill()
    _draw_voxels(ax, vg, m.vertices)

    # 2. point cloud
    ax = fig.add_subplot(1, 5, 2, projection="3d")
    pts, _ = trimesh_sample(m, 1400)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=F.ACCENT, depthshade=True)
    F._frame(ax, m.vertices)

    # 3. mesh (graph)
    ax = fig.add_subplot(1, 5, 3, projection="3d")
    F.add_mesh(ax, disp, base=F.MUTED, edge=F.INK, lw=0.15)

    # 4. SDF slice
    ax = fig.add_subplot(1, 5, 4)
    _sdf_slice(ax, m)

    # 5. B-rep schematic
    ax = fig.add_subplot(1, 5, 5)
    _brep_schematic(ax)

    for ax, t in zip(fig.axes, titles):
        ax.set_title(t, fontsize=11, color=F.INK, pad=2)
    fig.suptitle("One jet-engine bracket (SimJEB #148), five ways a network can read it",
                 y=1.02, fontsize=12.5)
    F.save(fig, "cae_spine_overview.png")


# ---------------------------------------------------------------- voxels
def fig_voxels():
    """Coarse vs fine voxelization -- the O(n^3) memory wall."""
    m = F.load_bracket()
    fig = plt.figure(figsize=(13, 6.2))
    for i, div in enumerate([14, 44]):
        pitch = m.extents.max() / div
        vg = m.voxelized(pitch=pitch).fill()
        occ = int(vg.matrix.sum())
        dense = int(np.prod(vg.matrix.shape))
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _draw_voxels(ax, vg, m.vertices)
        ax.set_title(f"pitch = {pitch:.1f} mm   grid {tuple(vg.matrix.shape)}\n"
                     f"{occ:,} occupied of {dense:,} cells "
                     f"({100*occ/dense:.1f}% full)", fontsize=10)
    fig.suptitle("Voxels: refining the grid multiplies memory as $O(n^3)$ "
                 "while the part stays mostly empty", y=1.0, fontsize=12)
    F.save(fig, "cae_voxels.png")


# ---------------------------------------------------------------- points
def fig_pointcloud():
    """Sparse vs dense surface sampling."""
    m = F.load_bracket()
    fig = plt.figure(figsize=(13, 6.2))
    for i, n in enumerate([256, 4096]):
        pts, _ = trimesh_sample(m, n)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6 if n < 1000 else 2.5,
                   c=F.ACCENT, depthshade=True)
        F._frame(ax, m.vertices)
        ax.set_title(f"$n = {n}$ points", fontsize=11)
    fig.suptitle("Point clouds: an unordered set on the surface; "
                 "compute scales linearly in $n$", y=1.0, fontsize=12)
    F.save(fig, "cae_pointcloud.png")


# ---------------------------------------------------------------- graph
def fig_graph():
    """A vertex and its 1-ring neighborhood -- message passing locality."""
    m = F.load_bracket()
    disp = F.decimate(m, grid=30)  # coarser so triangles are visible when zoomed
    adj = {i: set() for i in range(len(disp.vertices))}
    for a, b, c in disp.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u].add(v)
            adj[v].add(u)
    # pick a well-connected vertex to sit at the center of the zoom
    deg = np.array([len(adj[i]) for i in range(len(disp.vertices))])
    center = int(np.argsort(deg)[-30:][RNG.integers(0, 30)])
    ring1 = list(adj[center])
    ring2 = set()
    for r in ring1:
        ring2 |= adj[r]
    ring2 -= set(ring1) | {center}
    V = disp.vertices

    fig = plt.figure(figsize=(14, 6.0))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    F.add_mesh(ax, disp, base="#eef2f1", edge="#9fb2b0", lw=0.5, alpha=0.55)
    ax.scatter(*V[list(ring2)].T, s=26, c=F.MUTED, depthshade=False, zorder=5)
    ax.scatter(*V[ring1].T, s=70, c=F.ACCENT, depthshade=False,
               edgecolor="white", zorder=6)
    ax.scatter(*V[[center]].T, s=170, c=F.ACCENT2, depthshade=False,
               edgecolor="white", zorder=7)
    for r in ring1:
        seg = V[[center, r]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=F.ACCENT2, lw=2.2, zorder=6)
    # zoom the view to a box around the chosen vertex
    reach = np.linalg.norm(V[ring1] - V[center], axis=1).max() * 1.8
    c0 = V[center]
    ax.set_xlim(c0[0] - reach, c0[0] + reach)
    ax.set_ylim(c0[1] - reach, c0[1] + reach)
    ax.set_zlim(c0[2] - reach, c0[2] + reach)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(**F.VIEW)
    ax.set_axis_off()
    ax.set_title("mesh = graph: a node and its 1-ring", fontsize=11)

    ax2 = fig.add_subplot(1, 2, 2)
    _message_passing_schematic(ax2)
    fig.suptitle("Graphs: the FEA mesh is already a graph; "
                 "message passing aggregates over neighbors", y=1.0, fontsize=12)
    F.save(fig, "cae_graph.png")


# ---------------------------------------------------------------- sdf
def fig_sdf():
    """Signed-distance field: a 2D slice + the zero-level contour."""
    m = F.load_bracket()
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    _sdf_slice(ax, m, res=200, annotate=True)
    ax.set_title("Fields: signed distance on a $z$-slice through bracket #148\n"
                 "$s(\\mathbf{x})<0$ inside, $>0$ outside, "
                 "surface = zero level set", fontsize=11)
    F.save(fig, "cae_sdf.png")


# ---------------------------------------------------------------- brep
def fig_brep():
    """B-rep as a topological (face-adjacency) graph -- schematic."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    _brep_schematic(ax, big=True)
    ax.set_title("Programs: a CAD B-rep stores faces + edges + topology,\n"
                 "not a discretization", fontsize=11)
    F.save(fig, "cae_brep.png")


# ---------------------------------------------------------------- surrogate
def fig_surrogate():
    """The learning target: max stress vs geometry across 381 brackets."""
    import pandas as pd
    df = pd.read_csv(F.METADATA)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    cats = df["category"].astype("category")
    import matplotlib as mpl
    palette = mpl.colormaps["tab10"].resampled(len(cats.cat.categories))
    colors = [palette(i) for i in cats.cat.codes]

    for ax, (xcol, xlab) in zip(axes, [("mass", "mass"),
                                       ("surface_area", "surface area")]):
        ax.scatter(df[xcol], df["max_ver_stress"], s=22, c=colors,
                   alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("max von Mises stress (vertical load)")
        ax.grid(True, which="both", color=F.GRID, lw=0.6)
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", markersize=7,
                      markerfacecolor=palette(i), markeredgecolor="white",
                      label=c) for i, c in enumerate(cats.cat.categories)]
    axes[1].legend(handles=handles, fontsize=8, title="category",
                   loc="upper right", framealpha=0.9)
    fig.suptitle("The learning target: peak stress is a messy function of geometry "
                 "over 381 brackets", y=1.0, fontsize=12)
    F.save(fig, "cae_surrogate_target.png")


# ================================================================ helpers
def trimesh_sample(mesh, n):
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.asarray(pts), fid


def _draw_voxels(ax, vg, ref_pts):
    filled = vg.matrix
    # matplotlib voxels expects boolean grid; color occupied cells
    facecolors = np.empty(filled.shape + (4,), dtype=float)
    facecolors[..., :3] = np.array([0.16, 0.62, 0.56])  # teal
    facecolors[..., 3] = 0.95
    ax.voxels(filled, facecolors=facecolors, edgecolor=(1, 1, 1, 0.25),
              linewidth=0.2)
    try:
        ax.set_box_aspect(filled.shape, zoom=1.35)
    except TypeError:
        ax.set_box_aspect(filled.shape)
    ax.view_init(**F.VIEW)
    ax.set_axis_off()


def _sdf_slice(ax, mesh, res=140, annotate=False):
    """Compute SDF on a z-mid plane and draw filled contours."""
    import trimesh
    lo, hi = mesh.bounds
    zc = (lo[2] + hi[2]) / 2
    pad = 0.08 * (hi - lo).max()
    xs = np.linspace(lo[0] - pad, hi[0] + pad, res)
    ys = np.linspace(lo[1] - pad, hi[1] + pad, res)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, zc)])
    # signed distance: negative inside
    sd = -trimesh.proximity.signed_distance(mesh, P).reshape(X.shape)
    vmax = np.percentile(np.abs(sd), 96)
    im = ax.contourf(X, Y, sd, levels=24, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(X, Y, sd, levels=[0], colors=[F.INK], linewidths=1.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("signed distance $s(\\mathbf{x})$  (mm)")
        ax.text(0.02, 0.02, "black = surface  $s=0$", transform=ax.transAxes,
                fontsize=9, color=F.INK)


def _message_passing_schematic(ax):
    """Little diagram: neighbors -> message -> aggregate -> update."""
    ax.axis("off")
    center = (0.5, 0.5)
    nbrs = [(0.14, 0.80), (0.16, 0.28), (0.5, 0.90), (0.86, 0.74), (0.84, 0.22)]
    for p in nbrs:
        ax.annotate("", xy=center, xytext=p,
                    arrowprops=dict(arrowstyle="->", color=F.ACCENT2, lw=1.6))
        ax.add_patch(plt.Circle(p, 0.045, color=F.ACCENT, ec="white", zorder=3))
    ax.add_patch(plt.Circle(center, 0.06, color=F.ACCENT2, ec="white", zorder=4))
    ax.text(center[0], center[1] - 0.16, r"$h_i' = \phi(\,h_i,\ \bigoplus_{j\in N(i)}\ \psi(h_i,h_j)\,)$",
            ha="center", fontsize=12)
    ax.text(0.5, 1.02, "one message-passing step", ha="center", fontsize=10, color=F.INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)


def _brep_schematic(ax, big=False):
    """Face-adjacency graph of a simple prismatic solid (illustrative)."""
    import networkx as nx
    ax.axis("off")
    # a box with a through-hole: 6 outer faces + 1 cylindrical face
    G = nx.Graph()
    faces = ["top", "bottom", "front", "back", "left", "right", "hole"]
    G.add_nodes_from(faces)
    edges = [("top", "front"), ("top", "back"), ("top", "left"), ("top", "right"),
             ("bottom", "front"), ("bottom", "back"), ("bottom", "left"), ("bottom", "right"),
             ("front", "left"), ("front", "right"), ("back", "left"), ("back", "right"),
             ("hole", "top"), ("hole", "bottom")]
    G.add_edges_from(edges)
    pos = {"top": (0.5, 1.0), "bottom": (0.5, 0.0), "front": (0.15, 0.5),
           "back": (0.85, 0.5), "left": (0.32, 0.72), "right": (0.68, 0.72),
           "hole": (0.5, 0.5)}
    ncol = [F.ACCENT2 if f == "hole" else F.ACCENT for f in faces]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=F.INK, width=1.1, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncol, node_size=1100 if big else 380,
                           edgecolors="white")
    if big:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white")
    ax.text(0.5, -0.08, "nodes = faces,  edges = shared edges", ha="center",
            fontsize=9 if big else 8, color=F.INK, transform=ax.transAxes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.08)


# ================================================================ runner
FIGS = {
    "spine": fig_spine,
    "voxels": fig_voxels,
    "pointcloud": fig_pointcloud,
    "graph": fig_graph,
    "sdf": fig_sdf,
    "brep": fig_brep,
    "surrogate": fig_surrogate,
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for name in want:
        if name not in FIGS:
            print(f"  ?? unknown figure '{name}' (have: {', '.join(FIGS)})")
            continue
        print(f"[{name}]")
        FIGS[name]()
    print("done.")
