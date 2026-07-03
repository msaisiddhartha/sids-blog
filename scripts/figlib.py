"""Shared helpers for generating the figures in the
"3D Representations for CAE" blog post.

All figures are rendered from the SimJEB bracket dataset living OUTSIDE the
repo at ~/datasets/simjeb (see the post / project notes). Only the rendered
PNGs land in static/images/ and get committed -- never the raw meshes.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import trimesh  # noqa: E402

# ---------------------------------------------------------------- paths
DATA = os.path.expanduser("~/datasets/simjeb")
SAMPLE_OBJ = os.path.join(DATA, "sample", "148.obj")
SURFMESH_DIR = os.path.join(DATA, "surfmesh")
METADATA = os.path.join(DATA, "all_bracket_metadata.csv")

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.abspath(os.path.join(_HERE, "..", "static", "images"))

# ---------------------------------------------------------------- style
INK = "#2b2b2b"          # near-black line/text
ACCENT = "#2a9d8f"       # teal -- the post's accent color
ACCENT2 = "#e76f51"      # warm contrast (highlights)
MUTED = "#b8c4c2"        # light neutral surfaces
GRID = "#dfe6e5"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# consistent 3D camera so every panel shows the bracket the same way
VIEW = dict(elev=22, azim=-58)


def load_bracket(path=SAMPLE_OBJ):
    """Load a bracket, recenter to its centroid, return a trimesh."""
    m = trimesh.load(path, process=False)
    m.vertices = m.vertices - m.vertices.mean(axis=0)
    return m


def decimate(mesh, grid=64, target_faces=None):
    """Decimate a mesh for display.

    Prefers quadric decimation via `fast_simplification` (shape-preserving);
    falls back to cheap vertex clustering if it is unavailable. `target_faces`
    controls the quadric path; `grid` controls the clustering fallback. Either
    way the result is a display mesh, not a simulation mesh.
    """
    try:
        import fast_simplification
        if target_faces is None:
            # map the old grid knob onto a sensible face budget
            target_faces = int(np.clip(grid * grid * 4, 1500, 40000))
        target_faces = min(target_faces, len(mesh.faces))
        v, f = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
            target_count=target_faces)
        return trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f),
                               process=False)
    except Exception:
        pass  # fall back to vertex clustering below

    v = mesh.vertices
    lo, hi = v.min(0), v.max(0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    cell = np.floor((v - lo) / span * (grid - 1)).astype(np.int64)
    key = cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    # representative vertex = mean of members
    newv = np.zeros((len(uniq), 3))
    counts = np.zeros(len(uniq))
    np.add.at(newv, inv, v)
    np.add.at(counts, inv, 1.0)
    newv /= counts[:, None]
    newf = inv[mesh.faces]
    good = (newf[:, 0] != newf[:, 1]) & (newf[:, 1] != newf[:, 2]) & (newf[:, 0] != newf[:, 2])
    newf = newf[good]
    return trimesh.Trimesh(vertices=newv, faces=newf, process=False)


def shade_faces(mesh, base=MUTED, light=np.array([0.4, 0.5, 0.85])):
    """Return per-face RGBA with simple Lambertian shading for mpl."""
    from matplotlib.colors import to_rgb
    n = mesh.face_normals
    light = light / np.linalg.norm(light)
    intensity = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    rgb = np.array(to_rgb(base))
    cols = np.clip(intensity[:, None] * rgb[None, :], 0, 1)
    return np.hstack([cols, np.ones((len(cols), 1))])


def add_mesh(ax, mesh, base=MUTED, edge=None, lw=0.0, alpha=1.0):
    """Draw a shaded trimesh onto a 3D axis via Poly3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris = mesh.vertices[mesh.faces]
    coll = Poly3DCollection(tris, alpha=alpha)
    coll.set_facecolor(shade_faces(mesh, base))
    if edge is not None:
        coll.set_edgecolor(edge)
        coll.set_linewidth(lw)
    ax.add_collection3d(coll)
    _frame(ax, mesh.vertices)
    return coll


def _frame(ax, pts):
    """Equal aspect box + clean look for a 3D axis."""
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.4)  # zoom fills the panel
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(**VIEW)
    ax.set_axis_off()


def clean3d(fig):
    for ax in fig.axes:
        if hasattr(ax, "set_axis_off") and getattr(ax, "zaxis", None) is not None:
            ax.set_axis_off()


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote static/images/{name}  ({kb:.0f} KB)")
    return path
