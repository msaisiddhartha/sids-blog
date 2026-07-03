"""Animated 3D demonstration for the Fields section: sweeping the iso-level
of a bracket's signed-distance field so lightening holes open and merge --
a topology change the implicit field handles for free, with no remeshing.

Method (fast, no slow proximity queries):
  1. voxelize the SimJEB bracket into a solid occupancy grid,
  2. build a signed-distance grid with scipy's distance transform,
  3. march the surface at a sweep of levels L (skimage marching_cubes) --
     L<0 erodes (holes grow and merge), L>0 dilates,
  4. render each level from a fixed camera and write an animated GIF.

Usage:
    .venv/bin/python3 scripts/make_topology_gif.py [bracket_id]
    (default bracket 148 -- the post's running example)
"""
import os
import sys
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import figlib as F

VOX = 220          # voxels along the longest axis (finer = smoother)
PAD = 12           # background voxels around the part
SDF_BLUR = 1.7     # Gaussian sigma (voxels) applied to the SDF -> rounds off
                   # the voxel staircase before marching cubes
TAUBIN_ITERS = 14  # mesh smoothing passes (volume-preserving)
TARGET_FACES = 26000
FRAMES_EACH_WAY = 22
OUT = "cae_topology_sweep.gif"


def signed_distance_grid(mesh):
    """Voxelize -> solid occupancy -> smoothed signed distance grid (mm)."""
    pitch = mesh.extents.max() / VOX
    vg = mesh.voxelized(pitch=pitch).fill()
    occ = np.asarray(vg.matrix, dtype=bool)
    occ = np.pad(occ, PAD, mode="constant", constant_values=False)
    din = ndimage.distance_transform_edt(occ)        # + inside
    dout = ndimage.distance_transform_edt(~occ)       # + outside
    sdf = (dout - din) * pitch                        # >0 outside, <0 inside
    sdf = ndimage.gaussian_filter(sdf, sigma=SDF_BLUR * pitch / pitch)
    return sdf, pitch


def surface_at(sdf, level, pitch):
    """Marching cubes at `level` -> decimate -> Taubin smooth -> trimesh."""
    lo, hi = sdf.min(), sdf.max()
    if not (lo < level < hi):
        return None
    verts, faces, _, _ = measure.marching_cubes(sdf, level=level,
                                                spacing=(pitch, pitch, pitch))
    try:
        import fast_simplification
        verts, faces = fast_simplification.simplify(
            verts.astype(np.float32), faces.astype(np.int32),
            target_count=min(TARGET_FACES, len(faces)))
    except Exception:
        pass
    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                           process=False)
    # volume-preserving smoothing to erase residual facets / stair-steps
    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=TAUBIN_ITERS)
    except Exception:
        pass
    return mesh


def render(mesh, limits, level, eroding, size=(680, 560)):
    """Render one frame to a PIL image with a fixed camera + level readout."""
    fig = plt.figure(figsize=(size[0] / 130, size[1] / 130), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    if mesh is not None and len(mesh.faces):
        tris = mesh.vertices[mesh.faces]
        coll = Poly3DCollection(tris)
        coll.set_facecolor(F.shade_faces(mesh, base=F.ACCENT))
        coll.set_edgecolor("none")
        ax.add_collection3d(coll)
    (xl, yl, zl) = limits
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    try:
        ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]), zoom=1.7)
    except TypeError:
        ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]))
    ax.view_init(elev=26, azim=-60)
    ax.set_axis_off()
    tag = f"level set  s = {level:+.1f} mm"
    sub = "one field — holes open & merge as $s$ crosses zero"
    ax.text2D(0.5, 0.97, tag, transform=ax.transAxes, ha="center",
              fontsize=12, color=F.INK, weight="bold")
    ax.text2D(0.5, 0.925, sub, transform=ax.transAxes, ha="center",
              fontsize=10, color=F.ACCENT2)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return Image.fromarray(buf)


def main(bid=148):
    path = os.path.join(F.SURFMESH_DIR, f"{bid}.obj")
    if not os.path.exists(path):
        path = F.SAMPLE_OBJ
    print(f"[topology gif] bracket {bid}  ({path})")
    m = trimesh.load(path, process=False)
    m.vertices -= m.vertices.mean(0)
    sdf, pitch = signed_distance_grid(m)
    depth = -sdf.min()           # deepest interior distance (mm)
    print(f"  grid {sdf.shape}  pitch {pitch:.2f} mm  max interior depth {depth:.1f} mm")

    # Sweep a gentle band around zero: dilate slightly (holes shrink) down to
    # mild erosion (holes grow and neighbours merge) while the part stays
    # intact. Thin webs vanish quickly, so keep the erosion modest rather than
    # scaling with the thick-hub depth.
    hi_level = min(+2.5, 0.30 * depth)
    lo_level = max(-2.2, -0.30 * depth)
    down = np.linspace(hi_level, lo_level, FRAMES_EACH_WAY)

    # fixed camera limits from the most-dilated (largest) surface
    big = surface_at(sdf, hi_level, pitch)
    b = big.bounds
    pad = 0.06 * (b[1] - b[0]).max()
    limits = [(b[0][i] - pad, b[1][i] + pad) for i in range(3)]

    frames = []
    for L in down:
        mesh = surface_at(sdf, L, pitch)
        frames.append(render(mesh, limits, L, eroding=True))
        print(f"    L={L:+.2f}  faces={0 if mesh is None else len(mesh.faces)}")
    # hold the eroded frame, then play back to the filled state (breathing loop)
    seq = frames + [frames[-1]] * 4 + frames[::-1] + [frames[0]] * 4

    out = os.path.join(F.IMG_DIR, OUT)
    os.makedirs(F.IMG_DIR, exist_ok=True)
    seq[0].save(out, save_all=True, append_images=seq[1:], duration=80,
                loop=0, optimize=True)
    kb = os.path.getsize(out) / 1024
    print(f"  wrote static/images/{OUT}  ({kb:.0f} KB, {len(seq)} frames)")


if __name__ == "__main__":
    bid = int(sys.argv[1]) if len(sys.argv) > 1 else 148
    main(bid)
