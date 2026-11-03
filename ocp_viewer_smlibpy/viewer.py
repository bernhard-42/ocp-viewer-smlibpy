#
# Copyright 2026 Bernhard Walter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The smlibpy kernel plugin for ocp_tessellate: the tessellation contract
(steps 1 to 4) and the show contract (step 5) of ocp-tessellate's
ocp-tessellate's docs/ExternalKernel.md, plus `to_occt` for measuring.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ocp_tessellate import OcpGroup
from ocp_tessellate._kernel import binding as occt_binding, kernel
from ocp_tessellate.ocp_utils import BoundingBox
from ocp_tessellate.plugin import (
    CURVE_TYPES,
    SURFACE_TYPES,
    OcpExternalObject,
    external_coord_axis,
    external_coord_system,
    external_edges,
    external_scene,
    external_tessellate,
)
from ocp_tessellate.tessellator import compute_quality
from ocp_tessellate.types import ColorLike, DiscretizedEdges, Tessellation
from ocp_tessellate.utils import Color
from ocp_viewer_core.viewer import show
from smlibpy import (
    Axis,
    Edge,
    Face,
    Location,
    Mesh,
    Part,
    Plane,
    Shell,
    Sketch,
    Vector,
    Vertex,
    Wire,
    to_brep,
    to_smb,
)
from smlibpy._kernel import ext
from smlibpy.mesh import _triangle_faces
from smlibpy.shapes import _Body

__all__ = ["smlib_show", "smlib_show_all", "to_occt"]

# the kernel tag of smlib's bodies in the measurement backend's model (backend.py measures them)
KERNEL = "smlib"

# a body smlib tessellates: a part, a shell or a sketch (a face is shown as a shell of its own)
SmlibBody = _Body
# what smlib_show draws
SmlibDrawable = _Body | Face | Edge | Wire | Vertex | Vector | Location | Plane | Axis
# a kernel handle of smlibpy's _core (Brep, Edge, Curve): opaque here
KernelHandle = Any


# ===========================================================================
# The tessellation contract (ocp-tessellate's docs/ExternalKernel.md, section 2)
#   1. tessellate        the Tessellation dict of a body (step 1)
#   2. discretize_edges  the DiscretizedEdges dict of edges (step 2)
#   3a. bounding_box     the BoundingBox of a body (step 3a), a helper of convert
#   3b. cache_id         the cache key of a body (step 3b, optional), a helper of convert
#   4. convert           the orchestrator: one object into an OcpExternalObject or
#                        an OcpGroup (step 4), through ocp_tessellate's external_tessellate,
#                        external_edges, external_coord_system and external_coord_axis
# ===========================================================================


def tessellate(
    body: SmlibBody,
    quality: float,
    angular_tolerance: float,
    compute_edges: bool = True,
) -> Tessellation:
    """Step 1 of a kernel plugin: the Tessellation dict of a part, shell, sketch or
    face-as-shell (ocp-tessellate's docs/ExternalKernel.md 2.1): faces unwelded, one normal per
    vertex, edges as chord-sampled segments, the body's vertices.

    quality: the chord deviation in model units; angular_tolerance: in radians,
    the largest angle between neighbouring facet normals; compute_edges: whether
    to sample the edges (an empty edge set otherwise).
    """
    shape = body
    tolerance = float(quality)
    angle = float(np.degrees(angular_tolerance))
    max_edge = 0

    poly = ext.tessellate(
        shape._brep,
        chord_height_tolerance=float(tolerance),
        curve_angle_tolerance_deg=float(angle),
        surface_angle_tolerance_deg=float(angle),
        max_edge_length=float(max_edge),
    )
    mesh = Mesh._from_poly(poly, keep=True, brep=shape._brep)
    try:
        owner = _triangle_faces(shape, mesh)
        assert len(owner) == len(mesh.triangles)
        vertices, triangles, normals, triangles_per_face, face_ids = _unweld(
            mesh, owner
        )
    finally:
        mesh.dispose()

    _faces = shape._brep.faces()

    if compute_edges:
        edges, segments_per_edge, edge_types = _body_edge_segments(
            shape._brep, tolerance / 10
        )
    else:
        edges = np.zeros(0, dtype=np.float32)
        segments_per_edge = np.zeros(len(shape._brep.edges()), dtype=np.int32)
        edge_types = np.zeros(len(shape._brep.edges()), dtype=np.int32)

    return {
        # faces
        "vertices": vertices,
        "triangles": triangles,
        "normals": normals,
        "triangles_per_face": triangles_per_face,
        "face_types": np.asarray(
            [_get_face_type(_faces[fid]) for fid in face_ids], dtype=np.int32
        ),
        # edges
        "edges": edges,
        "segments_per_edge": segments_per_edge,
        "edge_types": edge_types,
        # vertices
        "obj_vertices": np.asarray(
            [tuple(v) for v in shape.vertices()], dtype=np.float32
        ).ravel(),
    }


def discretize_edges(
    edges: Iterable[Edge], deviation: float = 0.1, edge_accuracy: float | None = None
) -> DiscretizedEdges:
    """Step 2 of a kernel plugin: the DiscretizedEdges dict of smlib edges
    (ocp-tessellate's docs/ExternalKernel.md, step 2): point pairs per segment, segments per edge,
    edge types, and the end points as object vertices.

    The chord deviation is `edge_accuracy` when given, else a tenth of the quality
    of the edges' bounding box at `deviation` (the OCCT path's rule).
    """
    edges = list(edges)
    if edge_accuracy is None:
        lo = np.min([tuple(e.bbox().min) for e in edges], axis=0)
        hi = np.max([tuple(e.bbox().max) for e in edges], axis=0)
        bb = BoundingBox(
            {
                "xmin": lo[0],
                "xmax": hi[0],
                "ymin": lo[1],
                "ymax": hi[1],
                "zmin": lo[2],
                "zmax": hi[2],
            }
        )
        tolerance = compute_quality(bb, deviation=deviation) / 10
    else:
        tolerance = float(edge_accuracy)
    polylines, types, ends = [], [], []
    for edge in edges:
        kind = edge.geom_type
        if kind == "line":
            run = np.asarray([tuple(edge.start), tuple(edge.end)], dtype=np.float32)
        else:
            run = _sample_curve(
                lambda t, e=edge: tuple(e.point_at(float(t))), 0.0, 1.0, tolerance
            )
        polylines.append(run)
        types.append(_get_edge_type(kind))
        ends.extend([run[0], run[-1]])
    segments, per_edge = _segments(polylines)
    # the end points once each, as ocp_tessellate's discretize_edges keeps unique
    # vertices (shared corners of a wire appear once)
    ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    _, first = np.unique(np.round(ends, 6), axis=0, return_index=True)
    vertices = ends[np.sort(first)]
    return {
        "edges": segments,
        "segments_per_edge": per_edge,
        "edge_types": np.asarray(types, dtype=np.int32),
        "obj_vertices": vertices.ravel(),
    }


def bounding_box(body: SmlibBody) -> BoundingBox:
    """Step 3a of a kernel plugin: the body's box as ocp_tessellate's BoundingBox
    (ocp-tessellate's docs/ExternalKernel.md, step 3a): quality, camera and relative helper scales
    come from it. Used by `convert`."""
    bb = body.bbox()
    (xmin, ymin, zmin), (xmax, ymax, zmax) = tuple(bb.min), tuple(bb.max)
    return BoundingBox(
        {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }
    )


def cache_id(body: SmlibBody) -> str:
    """Step 3b of a kernel plugin (optional caching, ocp-tessellate's docs/ExternalKernel.md,
    step 3b): the body's key in the tessellation cache, a hash of its binary `.smb` bytes
    (3 ms for 180 faces). Equal for a body built twice, different for a moved copy. Measuring a
    body can store refitted trim curves in it (smlibpy's first area, volume or tight box), which
    changes the bytes: a cache miss, never a wrong mesh. Used by `convert`."""
    return hashlib.sha256(to_smb(body, ascii=False)).hexdigest()


def convert(
    obj: object, name: str, color: ColorLike | None = None, alpha: float | None = None
) -> OcpExternalObject | OcpGroup:
    """Step 4 of a kernel plugin, the orchestrator (ocp-tessellate's docs/ExternalKernel.md, step 4):
    one smlibpy object into what the show pipeline takes, an OcpExternalObject, or
    an OcpGroup of them for a list or dict.

    A part, shell, sketch or face becomes ocp_tessellate's `external_tessellate` with
    `tessellate` (cached by the cache id), the box and the cache id; an edge or
    wire `external_edges` with `discretize_edges`; a vertex or vector a leaf with
    `points`; a location or plane `external_coord_system`, an axis
    `external_coord_axis`. Without a color, a body's own `color` is used, else
    None and the pipeline fills in the host's default of the kind. A body is
    tagged with `KERNEL`, so the measurement backend measures it with backend.py.
    """
    if isinstance(obj, dict):
        return OcpGroup(
            [convert(v, str(k), color, alpha) for k, v in obj.items()], name=name
        )
    if isinstance(obj, (list, tuple)):
        return OcpGroup(
            [
                convert(c, f"{_group_name(c)}({i})", color, alpha)
                for i, c in enumerate(obj)
            ],
            name=name,
        )
    if isinstance(obj, Location):
        return external_coord_system(name, obj.position, obj.x_dir, obj.z_dir)
    if isinstance(obj, Plane):
        return external_coord_system(name, obj.origin, obj.x_dir, obj.z_dir)
    if isinstance(obj, Axis):
        return external_coord_axis(name, obj.origin, obj.direction, color)
    if isinstance(obj, (_Body, Face)):
        # a sketch and a face are drawn on both sides ("face"); a face is shown as
        # a shell of its own (Sketch is a Shell, so it is tested first)
        if isinstance(obj, Sketch):
            kind, body = "face", obj
        elif isinstance(obj, Part):
            kind, body = "solid", obj
        elif isinstance(obj, Shell):
            kind, body = "shell", obj
        elif isinstance(obj, Face):
            kind, body = "face", obj.to_shell()
        else:  # a _Body that is neither a Part nor a Shell: smlibpy has none
            raise TypeError(f"smlib_show: cannot show a {type(obj).__name__}")
        return external_tessellate(
            kind,
            name,
            body,
            tessellate,
            bounding_box(body),
            cache_id=cache_id(body),
            color=color,
            alpha=alpha,
            kernel=KERNEL,
        )
    if isinstance(obj, (Edge, Wire)):
        edges = obj.edges() if isinstance(obj, Wire) else [obj]
        return external_edges(name, edges, discretize_edges, color, alpha)
    if isinstance(obj, (Vertex, Vector)):
        return OcpExternalObject(
            "vertex",
            name,
            points=[tuple(obj)],
            color=None if color is None else Color(color),
        )
    raise TypeError(f"smlib_show: cannot show a {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Private helpers for the tessellation contract
# ---------------------------------------------------------------------------


# smlib's surface and curve kinds (``Face.geom_type``, ``Edge.geom_type`` or
# ``ext.curve_kind``) as the GeomAbs values the viewer expects: the face_types and
# edge_types of a mesh here, the geom_type names of the measurement backend (backend.py)
FACE_KIND_CODES: dict[str, int] = {
    "plane": SURFACE_TYPES["Plane"],
    "cylinder": SURFACE_TYPES["Cylinder"],
    "cone": SURFACE_TYPES["Cone"],
    "sphere": SURFACE_TYPES["Sphere"],
    "torus": SURFACE_TYPES["Torus"],
    "bspline": SURFACE_TYPES["BSplineSurface"],
    "revolution": SURFACE_TYPES["SurfaceOfRevolution"],
    "extrusion": SURFACE_TYPES["SurfaceOfExtrusion"],
    "offset": SURFACE_TYPES["OffsetSurface"],
    "other": SURFACE_TYPES["OtherSurface"],
}
EDGE_KIND_CODES: dict[str, int] = {
    "line": CURVE_TYPES["Line"],
    "circle": CURVE_TYPES["Circle"],
    "ellipse": CURVE_TYPES["Ellipse"],
    "bspline": CURVE_TYPES["BSplineCurve"],
    "other": CURVE_TYPES["OtherCurve"],
}


def _get_face_type(face: KernelHandle) -> int:
    """smlib's surface kind as the GeomAbs_SurfaceType value the viewer expects."""
    return FACE_KIND_CODES.get(Face(None, face).geom_type, SURFACE_TYPES["OtherSurface"])


def _get_edge_type(kind: str) -> int:
    """smlib's curve kind (``Edge.geom_type`` or ``ext.curve_kind``) as the
    GeomAbs_CurveType value the viewer expects."""
    return EDGE_KIND_CODES.get(kind, CURVE_TYPES["OtherCurve"])


def _group_name(child: object) -> str:
    for cls, label in (
        (Face, "Face"),
        (Edge, "Edge"),
        (Vertex, "Vertex"),
        (Wire, "Wire"),
    ):
        if isinstance(child, cls):
            return label
    return type(child).__name__


def _unweld(
    mesh: Mesh, owner: NDArray[np.int64]
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.int64],
]:
    """Lay the mesh out the way ocp-tessellate (and three.js) expects it.

    smlibpy's Mesh shares vertices between faces and keeps one normal per
    triangle corner (n, 3, 3). ocp-tessellate gives every face its own block of
    vertices (so a point on an edge shared by two faces appears once per face)
    and one normal per vertex. Returns flat vertices, flat triangle indices, flat
    normals, triangles_per_face and the face index of every block, blocks in
    shape.faces() order.
    """
    order = np.argsort(owner, kind="stable")
    tris = mesh.triangles[order]
    own = owner[order]
    assert mesh.normals is not None, "the kernel mesh has no corner normals"
    corner_normals = mesh.normals[order]  # (n, 3, 3), one per triangle corner

    face_ids = np.unique(own)
    starts = np.searchsorted(own, face_ids)
    stops = np.append(starts[1:], len(own))

    vertices, normals, triangles, triangles_per_face = [], [], [], []
    offset = 0
    for start, stop in zip(starts, stops):
        flat = tris[start:stop].ravel()
        corners = corner_normals[start:stop].reshape(-1, 3)
        # One vertex per distinct (point, normal) pair within the face. On a
        # smooth face all corners at a point share its surface normal, so this
        # is one vertex per point; where they differ (the apex of a cone, which
        # smlibpy keeps as one shared point) the point is split per normal, as
        # a per-vertex normal layout needs.
        keys = np.column_stack([flat, np.round(corners, 6)])
        _, first, local = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        local = local.ravel()
        vertices.append(mesh.vertices[flat[first]])
        normals.append(corners[first])
        triangles.append(local + offset)
        triangles_per_face.append(stop - start)
        offset += len(first)

    return (
        np.concatenate(vertices).astype(np.float32).ravel(),
        np.concatenate(triangles).astype(np.int32).ravel(),
        np.concatenate(normals).astype(np.float32).ravel(),
        np.asarray(triangles_per_face, dtype=np.int32),
        face_ids,
    )


def _body_edge_segments(
    brep: KernelHandle, tolerance: float
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    """The edges of a body (in ``brep.edges()`` order, seams included, as
    ``tessellate_boundaries`` lists them): segments, segments per edge, types."""
    polylines, types = [], []
    for handle in brep.edges():
        curve = handle.curve()
        kind = ext.curve_kind(curve)
        if kind == "line":
            # the vertices: exact, and no parameter evaluation at all
            run = np.asarray(
                [handle.start_vertex().point(), handle.end_vertex().point()],
                dtype=np.float32,
            )
        else:
            # the edge's interval can exceed the curve's natural range by a
            # rounding error (one ULP seen on a trimmed edge), which the kernel
            # rejects: clamp it
            t0, t1 = handle.parameter_range()
            c0, c1 = curve.parameter_range()
            run = _sample_curve(handle.point_at, max(t0, c0), min(t1, c1), tolerance)
        polylines.append(run)
        types.append(_get_edge_type(kind))
    edges, per_edge = _segments(polylines)
    return edges, per_edge, np.asarray(types, dtype=np.int32)


def _chord_deviation(
    pa: NDArray[np.float64], pm: NDArray[np.float64], pb: NDArray[np.float64]
) -> float:
    """Distance of the point pm from the chord pa-pb."""
    chord = pb - pa
    L2 = float(chord @ chord)
    if L2 == 0.0:
        return float(np.linalg.norm(pm - pa))
    u = float(np.clip((pm - pa) @ chord / L2, 0.0, 1.0))
    return float(np.linalg.norm(pm - pa - u * chord))


def _sample_curve(
    point_at: Callable[[float], Sequence[float]],
    t0: float,
    t1: float,
    tolerance: float,
    probes: int = 32,
    max_segments: int = 4096,
) -> NDArray[np.float32]:
    """Points along a curve whose chords stay within ``tolerance`` of it (as
    ocp_tessellate samples edges by deflection, at a tenth of the face quality).

    The segment count is estimated first: ``probes`` points along the curve give
    its length and local curvature per span, and a chord within deviation d of a
    circle of radius R may be 2 * sqrt(2 R d) long. The points are placed at that
    count, uniformly in the parameter; then every segment's midpoint is checked
    and the few segments the estimate missed (a tight bend between two probes)
    are bisected. ``point_at(t)`` evaluates the curve.
    """
    ts = np.linspace(t0, t1, probes + 1)
    pts = np.array([point_at(float(t)) for t in ts], dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    # curvature radius at the inner probes from the circumscribed circle of each
    # triple of consecutive points; the outer spans take their neighbour's
    a, b = pts[:-2], pts[1:-1]
    c = pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.linalg.norm(np.cross(ab, bc), axis=1)
    lens = (
        np.linalg.norm(ab, axis=1)
        * np.linalg.norm(bc, axis=1)
        * np.linalg.norm(ac, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(cross > 1e-15, lens / (2.0 * cross), np.inf)  # (probes - 1,)
    span_radius = np.minimum(
        np.r_[radius[0], radius], np.r_[radius, radius[-1]]
    )  # (probes,)
    # estimated for 80 % of the tolerance: segments placed exactly at the limit
    # would deviate by the tolerance within rounding, and half of them would be
    # bisected below, doubling the count where a tenth more would do
    margin = 0.8 * tolerance
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        max_chord = 2.0 * np.sqrt(
            np.maximum(2.0 * span_radius * margin - margin**2, 0.0)
        )
    need = np.where(
        np.isfinite(max_chord) & (max_chord > 0.0), seg_len / max_chord, 0.0
    )
    n = int(min(max(int(np.ceil(need.sum())), 1), max_segments))

    ts = np.linspace(t0, t1, n + 1)
    pts = [np.asarray(point_at(float(t)), dtype=np.float64) for t in ts]
    out = [pts[0]]
    stack = [(ts[i], pts[i], ts[i + 1], pts[i + 1], 0) for i in range(n)][::-1]
    while stack:
        ta, pa, tb, pb, depth = stack.pop()
        tm = 0.5 * (ta + tb)
        pm = np.asarray(point_at(float(tm)), dtype=np.float64)
        if _chord_deviation(pa, pm, pb) > tolerance and depth < 8:
            stack.append((tm, pm, tb, pb, depth + 1))
            stack.append((ta, pa, tm, pm, depth + 1))
        else:
            out.append(pb)
    return np.asarray(out, dtype=np.float32)


def _segments(
    polylines: Sequence[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Point pairs per segment and the segment count per polyline."""
    pairs = [
        np.stack([run[:-1], run[1:]], axis=1).ravel()
        for run in polylines
        if len(run) >= 2
    ]
    counts = np.asarray([max(len(run) - 1, 0) for run in polylines], dtype=np.int32)
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float32), counts
    return np.concatenate(pairs).astype(np.float32), counts


# ===========================================================================
# The show contract (ocp-tessellate's docs/ExternalKernel.md, section 3)
#   5. smlib_show, smlib_show_all  the exported functions (step 5): the scene of
#                        ocp_tessellate's external_scene(objects, convert, ...) 
#                        into the host's show
# ===========================================================================


def smlib_show(
    *objects: object,
    names: Sequence[str | None] | None = None,
    colors: Sequence[ColorLike | None] | None = None,
    alphas: Sequence[float | None] | None = None,
    **kwargs: Any,
) -> Any:
    """Step 5 of a kernel plugin (ocp-tessellate's docs/ExternalKernel.md, step 5): show smlibpy objects,
    tessellated by smlib's kernel, in the viewer of the host.

    The scene of ocp_tessellate's `external_scene` (every object through `convert`)
    goes to the host's `show`
    (`ocp_viewer_core.viewer.show`: ocp_vscode, jupyter_cadquery, build123d_studio
    or ocp_viewer, picked at import); every keyword of `show` applies
    (`reset_camera`, `collapse`, `helper_scale`, ...) and so do `set_defaults`.
    Bodies are cached in ocp_tessellate's tessellation cache by their geometry,
    and measured by smlib's geometry (backend.py).
    """
    scene, names, colors, alphas = external_scene(
        objects, convert, names, colors, alphas
    )
    return show(*scene, names=names, colors=colors, alphas=alphas, **kwargs)


def smlib_show_all(
    variables: Mapping[str, object] | None = None,
    exclude: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """``smlib_show`` of every smlibpy object in the caller's scope (as
    the hosts' ``show_all``), the variable names as object names."""
    if variables is None:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        variables = frame.f_back.f_locals
    exclude = list(exclude or []) + ["_", "__", "___", "_ih", "_oh", "_dh", "Out", "In"]
    include = list(include or [])
    objects, names = [], []
    for name, obj in variables.items():
        if (
            name in exclude
            or name.startswith("__")
            or re.match(r"^_i?\d+", name) is not None
        ):
            continue
        if name in include or _is_smlib_drawable(obj):
            objects.append(obj)
            names.append(name)
    if len(objects) == 0:
        print("smlib_show_all: no smlibpy objects in scope")
        return None
    return smlib_show(*objects, names=names, **kwargs)


# ---------------------------------------------------------------------------
# Private helpers for the show contract
# ---------------------------------------------------------------------------


def _is_smlib_drawable(obj: object) -> bool:
    """What smlib_show can draw: smlibpy bodies, faces, edges, wires, vertices,
    vectors, and lists or dicts holding any of them."""
    if isinstance(obj, dict):
        return len(obj) > 0 and any(_is_smlib_drawable(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return len(obj) > 0 and any(_is_smlib_drawable(v) for v in obj)
    return isinstance(
        obj, (_Body, Face, Edge, Wire, Vertex, Vector, Location, Plane, Axis)
    )


# ===========================================================================
# Helpers
#
#   to_occt              the OCCT conversion, can be used for measuring
#                        as in show(to_occt(smlib_obj)) with hosts show
# ===========================================================================


def to_occt(obj: object, binding: str | None = None) -> Any:
    """OCCT objects for smlibpy objects, so that ``show(to_occt(obj))`` measures
    like any OCCT object: parts, shells, sketches and faces through smlib's
    ``.brep`` writer, edges and wires as exact lines or NURBS curves, vertices
    and vectors as vertices, locations as ``TopLoc_Location``, planes as ``gp_Pln``, axes
    as ``gp_Ax1``; lists and dicts element-wise.

    ``binding`` is ``"OCP"`` or ``"nanocct"``; by default ocp_tessellate's
    ``kernel()``, nanocct where it is installed, else OCP. Raises ``ValueError``
    where smlib cannot write a body (offset surfaces of free-form faces): show
    those with ``smlib_show``.
    """
    occ = kernel() if binding is None else occt_binding(binding)
    pnt_array, real_array, int_array = _occt_arrays(occ)

    def pnt(v: Any) -> Any:
        return occ.gp.gp_Pnt(*tuple(v))

    def array(cls: Any, values: Iterable[Any], conv: Callable[[Any], Any]) -> Any:
        values = list(values)
        a = cls(1, len(values))
        for i, v in enumerate(values, 1):
            a.SetValue(i, conv(v))
        return a

    def edge(e: Edge) -> Any:
        if e.geom_type == "line":
            return occ.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(
                pnt(e.start), pnt(e.end)
            ).Edge()
        n = e.nurbs()
        uniq, mult = np.unique(np.round(n.knots, 12), return_counts=True)
        poles = array(pnt_array, n.points, pnt)
        knots = array(real_array, uniq, float)
        mults = array(int_array, mult, int)
        if n.weights is None:
            curve = occ.Geom.Geom_BSplineCurve(poles, knots, mults, n.degree)
        else:
            curve = occ.Geom.Geom_BSplineCurve(
                poles, array(real_array, n.weights, float), knots, mults, n.degree
            )
        return occ.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(curve).Edge()

    def location(loc: Location) -> Any:
        m = np.asarray(loc.matrix, dtype=np.float64)
        t = occ.gp.gp_Trsf()
        t.SetValues(*m[:3].ravel().tolist())
        return occ.TopLoc.TopLoc_Location(t)

    def body(b: _Body | Face) -> Any:
        try:
            text = to_brep(b)
        except (
            Exception
        ) as ex:  # KernelError for offset surfaces, ValueError for empty bodies
            raise ValueError(
                f"to_occt: smlib cannot write this {type(b).__name__} as .brep ({ex}); "
                "show it with smlib_show instead"
            ) from ex
        shape = occ.TopoDS.TopoDS_Shape()
        # OCP's Read_s reads a BytesIO, nanocct's a text stream
        stream = (
            io.StringIO(text)
            if occ.__name__ == "nanocct"
            else io.BytesIO(text.encode("ascii"))
        )
        occ.BRepTools.BRepTools.Read_s(shape, stream, occ.BRep.BRep_Builder())
        if shape.IsNull():
            raise ValueError("to_occt: OCCT read no shape from smlib's .brep text")
        return shape

    def convert(o: object) -> Any:
        if isinstance(o, dict):
            return {key: convert(v) for key, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        if isinstance(o, (_Body, Face)):
            return body(o)
        if isinstance(o, Wire):
            maker = occ.BRepBuilderAPI.BRepBuilderAPI_MakeWire()
            for e in o.edges():
                maker.Add(edge(e))
            return maker.Wire()
        if isinstance(o, Edge):
            return edge(o)
        if isinstance(o, (Vertex, Vector)):
            return occ.BRepBuilderAPI.BRepBuilderAPI_MakeVertex(pnt(o)).Vertex()
        if isinstance(o, Plane):
            ax3 = occ.gp.gp_Ax3(
                pnt(o.origin),
                occ.gp.gp_Dir(*tuple(o.z_dir)),
                occ.gp.gp_Dir(*tuple(o.x_dir)),
            )
            return occ.gp.gp_Pln(ax3)
        if isinstance(o, Location):
            return location(o)
        if isinstance(o, Axis):
            return occ.gp.gp_Ax1(pnt(o.origin), occ.gp.gp_Dir(*tuple(o.direction)))
        raise TypeError(f"to_occt: cannot convert a {type(o).__name__}")

    return convert(obj)


def _occt_arrays(occ: ModuleType) -> tuple[Any, Any, Any]:
    """The Array1 classes for points, reals and integers of the binding `occ`:
    nanocct's NCollection templates, OCP 7's TColgp/TColStd classes, OCP 8's
    OCP.collections (the split ocp_tessellate's OCP adapter makes)."""
    if occ.__name__ == "nanocct":
        return (
            occ.NCollection.NCollection_Array1__gp_Pnt,
            occ.NCollection.NCollection_Array1__double,
            occ.NCollection.NCollection_Array1__int,
        )
    if occ.__version__.startswith("7"):
        return (
            occ.TColgp.TColgp_Array1OfPnt,
            occ.TColStd.TColStd_Array1OfReal,
            occ.TColStd.TColStd_Array1OfInteger,
        )
    return (
        occ.collections.Array1_gp_Pnt,
        occ.collections.Array1_double,
        occ.collections.Array1_int,
    )
