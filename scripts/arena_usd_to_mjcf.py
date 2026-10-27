"""Convert the ARENA2036 shopfloor (arena_no_walls.usd from robotics-lab/imported_cad) to MuJoCo.

Writes the visual meshes arena_vis_*.obj (merged per colour, contype=0, small parts dropped) into
--out (xmls/arena/), and next to it xmls/arena.xml, the <mujoco> include with the assets, the world
geoms and, inline, the collision heightfield of everything between the floor and --hfield-top.
arena.xml has to sit in xmls/: MuJoCo resolves asset files of an included file relative to the
include's directory.

The hall's floor slab is not exported: the scene's floor plane is the floor. All coordinates are
shifted by --offset in x/y (the same shift scene_kuka.xml applies to the KUKA cell, so cell and hall
keep the relative pose they have in robotics-lab's scene.usd) and so that the slab top is z = 0.

Fences are single meshes (a 7.7 x 7.3 m cell fence is one connected mesh), so boxes or convex hulls
would fill the cells. The heightfield samples the surfaces instead; anything overhanging (tables)
becomes a column down to the floor.

Needs pxr (usd-core), numpy and fast-simplification (only for --decimate > 0). Run with:
  python3 scripts/arena_usd_to_mjcf.py --usd .../arena_no_walls.usd
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "src", "assets", "robots", "unitree_r1", "xmls", "arena")
# body pos of imported_kuka_assembly in scene_kuka.xml (x, y)
KUKA_OFFSET = (10.244803, 14.807147)


def triangulate(mesh):
  """World-space triangles (N,3,3) and per-corner normals (N,3,3) or None."""
  counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
  idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
  pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
  if len(counts) == 0 or len(pts) == 0:
    return None, None
  starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
  # fan triangulation: corner (start, start+k, start+k+1) for k in 1..count-2
  ntri = np.maximum(counts - 2, 0)
  face = np.repeat(np.arange(len(counts)), ntri)
  k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
  corners = np.stack([starts[face], starts[face] + k, starts[face] + k + 1], axis=1)
  tris = pts[idx[corners]]

  normals = None
  attr = mesh.GetNormalsAttr()
  n = attr.Get() if attr.HasAuthoredValue() else None
  if n is not None and len(n):
    n = np.asarray(n, dtype=np.float64)
    interp = mesh.GetNormalsInterpolation()
    if interp == UsdGeom.Tokens.faceVarying and len(n) == len(idx):
      normals = n[corners]
    elif interp == UsdGeom.Tokens.vertex and len(n) == len(pts):
      normals = n[idx[corners]]
  return tris, normals


def colour(mesh_prim, cache):
  mat, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
  if not mat:
    return (0.7, 0.7, 0.7, 1.0)
  key = mat.GetPath()
  if key not in cache:
    rgb, alpha = (0.7, 0.7, 0.7), 1.0
    for p in Usd.PrimRange(mat.GetPrim(), Usd.TraverseInstanceProxies()):
      if p.IsA(UsdShade.Shader):
        sh = UsdShade.Shader(p)
        c = sh.GetInput("diffuseColor") or sh.GetInput("diffuse_color_constant")
        if c and c.Get() is not None:
          rgb = tuple(float(x) for x in c.Get())
        o = sh.GetInput("opacity")
        if o and o.Get() is not None:
          alpha = float(o.Get())
        break
    cache[key] = tuple(round(x, 3) for x in (*rgb, alpha))
  return cache[key]


def face_normals(tris):
  """Per-corner normals (N,3,3) of flat-shaded triangles."""
  fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
  return np.repeat(fn[:, None, :], 3, axis=1)


def decimate(tris, reduction):
  """Quadric decimation of one part (welded by position first); the result is flat-shaded."""
  import fast_simplification  # only needed with --decimate

  v, f = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0, return_inverse=True)
  f = f.reshape(-1, 3)
  f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
  v2, f2 = fast_simplification.simplify(v, f, target_reduction=reduction)
  return v2[f2] if len(f2) else tris


def write_obj(path, tris, normals):
  """Positions and normals indexed separately per corner, so CAD hard edges keep their normals."""
  vn = normals.reshape(-1, 3)
  vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
  v, vi = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
  vn, ni = np.unique(np.round(vn, 3), axis=0, return_inverse=True)
  fi = np.stack([vi.ravel() + 1, ni.ravel() + 1], axis=1).reshape(-1, 6)
  with open(path, "w") as f:
    f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v))
    f.write("".join(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in vn))
    f.write("".join(f"f {a}//{b} {c}//{d} {e}//{g}\n" for a, b, c, d, e, g in fi))


def sample_surface(tris, step):
  """Points on each triangle on a barycentric grid with spacing <= step."""
  edge = np.max(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2), axis=1)
  nseg = np.maximum(np.ceil(edge / step).astype(int), 1)
  out = []
  for n in np.unique(nseg):
    t = tris[nseg == n]
    a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = a + b <= n
    w = np.stack([a[keep], b[keep]], axis=1) / n  # (m,2) barycentric of corners 1, 2
    out.append(
      (t[:, None, 0] + w[None, :, :1] * (t[:, None, 1] - t[:, None, 0])
       + w[None, :, 1:] * (t[:, None, 2] - t[:, None, 0])).reshape(-1, 3)
    )
  return np.concatenate(out) if out else np.zeros((0, 3))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--usd", required=True, help="arena_no_walls.usd")
  ap.add_argument("--out", default=DEFAULT_OUT)
  ap.add_argument("--offset", type=float, nargs=2, default=KUKA_OFFSET, metavar=("X", "Y"))
  ap.add_argument("--min-size", type=float, default=0.10, help="drop visual parts smaller than this (m)")
  ap.add_argument("--chunk", type=int, default=150_000, help="max triangles per visual mesh")
  # Every head camera view draws the whole hall (MuJoCo does not cull), so its triangles cost
  # render time four times over in stereo. Needs fast-simplification.
  # 0.75: 1.17 M -> 0.41 M triangles, stereo head camera 23 -> 27 fps on the MX150 laptop.
  ap.add_argument("--decimate", type=float, default=0.75,
                  help="remove this share of the triangles of each visual part (0 = off)")
  ap.add_argument("--decimate-min", type=int, default=300, help="leave parts with fewer triangles alone")
  ap.add_argument("--cell", type=float, default=0.04, help="heightfield cell size (m)")
  ap.add_argument("--hfield-min", type=float, default=0.03, help="ignore geometry below this height (m)")
  ap.add_argument("--hfield-top", type=float, default=1.8, help="ignore geometry above this height (m)")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  stage = Usd.Stage.Open(args.usd)
  assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdGeom.GetStageUpAxis(stage) == "Z"
  xf = UsdGeom.XformCache()
  colours = {}
  parts = []  # (path, rgba, tris, normals)
  for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):
      continue
    tris, normals = triangulate(UsdGeom.Mesh(prim))
    if tris is None:
      continue
    m = np.asarray(xf.GetLocalToWorldTransform(prim))
    tris = tris @ m[:3, :3] + m[3, :3]
    if normals is not None:
      normals = normals @ np.linalg.inv(m[:3, :3]).T
    parts.append((str(prim.GetPath()), colour(prim, colours), tris, normals))
  print(f"[usd] {len(parts)} meshes, {sum(len(p[2]) for p in parts)} triangles")

  # The floor slab: the part with the largest footprint.
  def footprint(t):
    s = np.ptp(t.reshape(-1, 3), axis=0)
    return s[0] * s[1]

  slab = max(range(len(parts)), key=lambda i: footprint(parts[i][2]))
  floor = float(parts[slab][2][..., 2].max())
  print(f"[usd] floor slab {parts[slab][0]}: top at z = {floor:.6f}")
  shift = np.array([args.offset[0], args.offset[1], -floor])
  del parts[slab]

  # ---- visuals: merge per colour, chunked ----
  groups = defaultdict(list)
  dropped = 0
  for path, rgba, tris, normals in parts:
    if np.ptp(tris.reshape(-1, 3), axis=0).max() < args.min_size:
      dropped += len(tris)
      continue
    if args.decimate > 0 and len(tris) >= args.decimate_min:
      tris, normals = decimate(tris, args.decimate), None
    groups[rgba].append((tris + shift, face_normals(tris) if normals is None else normals))
  print(f"[vis] dropped {dropped} triangles of parts < {args.min_size} m")
  print(f"[vis] {sum(len(t) for g in groups.values() for t, _ in g)} triangles left")

  assets, geoms = [], []
  for ci, (rgba, items) in enumerate(sorted(groups.items(), key=lambda kv: -sum(len(t) for t, _ in kv[1]))):
    mat = f"arena_mat_{ci}"
    assets.append(f'    <material name="{mat}" rgba="{" ".join(f"{x:g}" for x in rgba)}" />')
    chunk, count, part = [], 0, 0
    for t, n in items + [(None, None)]:
      if t is not None:
        chunk.append((t, n))
        count += len(t)
      if chunk and (t is None or count >= args.chunk):
        tris = np.concatenate([c[0] for c in chunk])
        norms = np.concatenate([c[1] for c in chunk])
        name = f"arena_vis_{ci}_{part}"
        write_obj(os.path.join(args.out, name + ".obj"), tris, norms)
        assets.append(f'    <mesh name="{name}" file="../arena/{name}.obj" inertia="shell" />')
        geoms.append(
          f'    <geom name="{name}" type="mesh" mesh="{name}" material="{mat}" '
          f'contype="0" conaffinity="0" group="1" />'
        )
        print(f"[vis] {name}: {len(tris)} triangles, rgba {rgba}")
        chunk, count, part = [], 0, part + 1

  # ---- collision heightfield ----
  lo_z, hi_z = floor + args.hfield_min, floor + args.hfield_top
  pts = []
  for _, _, tris, _ in parts:
    z = tris[..., 2]
    t = tris[(z.max(axis=1) > lo_z) & (z.min(axis=1) <= hi_z)]
    if len(t):
      p = sample_surface(t, args.cell / 2)
      pts.append(p[(p[:, 2] > lo_z) & (p[:, 2] <= hi_z)])
  pts = np.concatenate(pts) + shift
  allv = np.concatenate([p[2].reshape(-1, 3) for p in parts]) + shift
  x0, y0 = allv[:, 0].min() - 0.5, allv[:, 1].min() - 0.5
  x1, y1 = allv[:, 0].max() + 0.5, allv[:, 1].max() + 0.5
  ncol = int(np.ceil((x1 - x0) / args.cell)) + 1
  nrow = int(np.ceil((y1 - y0) / args.cell)) + 1
  h = np.zeros((nrow, ncol), dtype=np.float32)
  c = np.clip(np.round((pts[:, 0] - x0) / args.cell).astype(int), 0, ncol - 1)
  r = np.clip(np.round((pts[:, 1] - y0) / args.cell).astype(int), 0, nrow - 1)
  np.maximum.at(h, (r, c), pts[:, 2].astype(np.float32))
  hmax = float(h.max())
  rx, ry = (ncol - 1) * args.cell / 2, (nrow - 1) * args.cell / 2
  print(f"[hfield] {nrow} x {ncol} cells of {args.cell} m, {int((h > 0).sum())} occupied, max {hmax:.2f} m")

  # Inline elevation instead of a file: MuJoCo 3.3.6 (simulate/) resolves hfield files relative to
  # the model, newer versions relative to meshdir, so no relative path works for both.
  # elevation starts at +y (unlike the binary file format), column 0 at -x; values in [0, 1],
  # scaled by size[2] = hmax.
  elevation = " ".join("0" if v == 0 else f"{v:.4g}" for v in (h[::-1] / hmax).ravel())
  assets.append(
    f'    <hfield name="arena_hfield" nrow="{nrow}" ncol="{ncol}" '
    f'size="{rx:.4f} {ry:.4f} {hmax:.4f} 0.05" elevation="{elevation}" />'
  )
  # 1 cm below the floor plane: free cells (height 0) never touch the feet, the plane carries them.
  geoms.append(
    f'    <geom name="arena_collision" type="hfield" hfield="arena_hfield" '
    f'pos="{x0 + rx:.4f} {y0 + ry:.4f} -0.01" group="3" rgba="0.9 0.3 0.3 0.4" />'
  )

  xml = os.path.join(args.out, "..", "arena.xml")
  with open(xml, "w") as f:
    f.write(
      "<!-- Generated by scripts/arena_usd_to_mjcf.py from arena_no_walls.usd; do not edit. -->\n"
      '<mujoco model="arena2036_shopfloor">\n  <asset>\n'
      + "\n".join(assets)
      + "\n  </asset>\n  <worldbody>\n"
      + "\n".join(geoms)
      + "\n  </worldbody>\n</mujoco>\n"
    )
  print(f"[out] {os.path.normpath(xml)}")


if __name__ == "__main__":
  main()
