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

"""smlib's implementation of ocp_tessellate's TessellationProtocol: what makes smlibpy's
shapes visible in the viewer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from ocp_tessellate.ocp_utils import BoundingBox
from ocp_tessellate.plugin import CURVE_TYPES, SURFACE_TYPES, TessellationProtocol
from ocp_tessellate.types import ConvertedVertices, DiscretizedEdges, Tessellation
from smlibpy import Edge, Face, Mesh, Vector, Vertex, to_smb
from smlibpy._kernel import ext
from smlibpy.mesh import _triangle_faces
from smlibpy.shapes import _Body

from .types import KernelHandle

__all__ = ["EDGE_KIND_CODES", "FACE_KIND_CODES", "SmlibTessellation"]

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


class SmlibTessellation(TessellationProtocol[_Body, Edge, Vertex | Vector]):
    def tessellate(
        self,
        shape: _Body,
        deviation: float,
        quality: float,
        angular_tolerance: float,
        compute_edges: bool = True,
        compute_uvs: bool = False,
        normalize_uvs: bool = True,
        shape_id: str = "",
        debug: bool = False,
    ) -> Tessellation:
        """The mesh of a part, shell, sketch or face as a shell: faces unwelded, one normal
        per vertex, edges as chord-sampled segments, the body's vertices. The quality is
        the chord deviation in model units, the angular tolerance in radians the largest
        angle between neighbouring facet normals. smlib has no materials, so the UV
        arguments are ignored."""
        tolerance = float(quality)
        angle = float(np.degrees(angular_tolerance))

        poly = ext.tessellate(
            shape._brep,
            chord_height_tolerance=tolerance,
            curve_angle_tolerance_deg=angle,
            surface_angle_tolerance_deg=angle,
            max_edge_length=0.0,
        )
        mesh = Mesh._from_poly(poly, keep=True, brep=shape._brep)
        try:
            owner = _triangle_faces(shape, mesh)
            assert len(owner) == len(mesh.triangles)
            vertices, triangles, normals, triangles_per_face, face_ids = self._unweld(mesh, owner)
        finally:
            mesh.dispose()

        faces = shape._brep.faces()

        if compute_edges:
            edges, segments_per_edge, edge_types = self._body_edge_segments(shape._brep, tolerance / 10)
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
            "face_types": np.asarray([self._get_face_type(faces[fid]) for fid in face_ids], dtype=np.int32),
            # edges
            "edges": edges,
            "segments_per_edge": segments_per_edge,
            "edge_types": edge_types,
            # vertices
            "obj_vertices": np.asarray([tuple(v) for v in shape.vertices()], dtype=np.float32).ravel(),
        }

    def discretize_edges(self, edges: Iterable[Edge], deflection: float = 0.1, shape_id: str = "") -> DiscretizedEdges:
        """Point pairs per segment, segments per edge, edge types, and the end points as
        object vertices. The chord deviation is `deflection`."""
        tolerance = float(deflection)
        polylines, types, ends = [], [], []
        for edge in edges:
            kind = edge.geom_type
            if kind == "line":
                run = np.asarray([tuple(edge.start), tuple(edge.end)], dtype=np.float32)
            else:
                run = self._sample_curve(lambda t, e=edge: tuple(e.point_at(float(t))), 0.0, 1.0, tolerance)
            polylines.append(run)
            types.append(self._get_edge_type(kind))
            ends.extend([run[0], run[-1]])
        segments, per_edge = self._segments(polylines)
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

    def convert_vertices(self, vertices: Iterable[Vertex | Vector], shape_id: str = "") -> ConvertedVertices:
        """The coordinates of vertices or vectors."""
        return {"obj_vertices": np.asarray([tuple(v) for v in vertices], dtype=np.float32).ravel()}

    def bounding_box(
        self, shape: _Body | Face | Edge | Vertex | Vector | Sequence[Edge | Vertex | Vector]
    ) -> BoundingBox:
        """The box of a shape, or a list of edges or vertices. A location, plane or axis
        has no extent and raises, which adds nothing to a box."""
        items = shape if isinstance(shape, Sequence) else [shape]
        lows, highs = zip(*(self._box_corners(item) for item in items))
        (xmin, ymin, zmin), (xmax, ymax, zmax) = np.min(lows, axis=0), np.max(highs, axis=0)
        return BoundingBox(
            {
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "zmin": float(zmin),
                "zmax": float(zmax),
            }
        )

    def cache_id(self, shape: _Body) -> str:
        """A hash of the body's binary `.smb` bytes (3 ms for 180 faces), which hold neither
        name nor color: equal for a body built twice, different for a moved copy. Measuring
        a body can store refitted trim curves in it (smlibpy's first area, volume or tight
        box), which changes the bytes: a cache miss, never a wrong mesh."""
        return hashlib.sha256(to_smb(shape, ascii=False)).hexdigest()

    def _box_corners(self, shape: _Body | Face | Edge | Vertex | Vector) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if isinstance(shape, Vector):
            return tuple(shape), tuple(shape)
        bb = shape.bbox()
        return tuple(bb.min), tuple(bb.max)

    def _get_face_type(self, face: KernelHandle) -> int:
        """smlib's surface kind as the GeomAbs_SurfaceType value the viewer expects."""
        return FACE_KIND_CODES.get(Face(None, face).geom_type, SURFACE_TYPES["OtherSurface"])

    def _get_edge_type(self, kind: str) -> int:
        """smlib's curve kind (``Edge.geom_type`` or ``ext.curve_kind``) as the
        GeomAbs_CurveType value the viewer expects."""
        return EDGE_KIND_CODES.get(kind, CURVE_TYPES["OtherCurve"])

    def _unweld(
        self, mesh: Mesh, owner: NDArray[np.int64]
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
            _, first, local = np.unique(keys, axis=0, return_index=True, return_inverse=True)
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
        self, brep: KernelHandle, tolerance: float
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
                run = self._sample_curve(handle.point_at, max(t0, c0), min(t1, c1), tolerance)
            polylines.append(run)
            types.append(self._get_edge_type(kind))
        edges, per_edge = self._segments(polylines)
        return edges, per_edge, np.asarray(types, dtype=np.int32)

    def _chord_deviation(self, pa: NDArray[np.float64], pm: NDArray[np.float64], pb: NDArray[np.float64]) -> float:
        """Distance of the point pm from the chord pa-pb."""
        chord = pb - pa
        L2 = float(chord @ chord)
        if L2 == 0.0:
            return float(np.linalg.norm(pm - pa))
        u = float(np.clip((pm - pa) @ chord / L2, 0.0, 1.0))
        return float(np.linalg.norm(pm - pa - u * chord))

    def _sample_curve(
        self,
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
        lens = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(cross > 1e-15, lens / (2.0 * cross), np.inf)  # (probes - 1,)
        span_radius = np.minimum(np.r_[radius[0], radius], np.r_[radius, radius[-1]])  # (probes,)
        # estimated for 80 % of the tolerance: segments placed exactly at the limit
        # would deviate by the tolerance within rounding, and half of them would be
        # bisected below, doubling the count where a tenth more would do
        margin = 0.8 * tolerance
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            max_chord = 2.0 * np.sqrt(np.maximum(2.0 * span_radius * margin - margin**2, 0.0))
        need = np.where(np.isfinite(max_chord) & (max_chord > 0.0), seg_len / max_chord, 0.0)
        n = int(min(max(int(np.ceil(need.sum())), 1), max_segments))

        ts = np.linspace(t0, t1, n + 1)
        pts = [np.asarray(point_at(float(t)), dtype=np.float64) for t in ts]
        out = [pts[0]]
        stack = [(ts[i], pts[i], ts[i + 1], pts[i + 1], 0) for i in range(n)][::-1]
        while stack:
            ta, pa, tb, pb, depth = stack.pop()
            tm = 0.5 * (ta + tb)
            pm = np.asarray(point_at(float(tm)), dtype=np.float64)
            if self._chord_deviation(pa, pm, pb) > tolerance and depth < 8:
                stack.append((tm, pm, tb, pb, depth + 1))
                stack.append((ta, pa, tm, pm, depth + 1))
            else:
                out.append(pb)
        return np.asarray(out, dtype=np.float32)

    def _segments(
        self,
        polylines: Sequence[NDArray[np.float32]],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Point pairs per segment and the segment count per polyline."""
        pairs = [np.stack([run[:-1], run[1:]], axis=1).ravel() for run in polylines if len(run) >= 2]
        counts = np.asarray([max(len(run) - 1, 0) for run in polylines], dtype=np.int32)
        if len(pairs) == 0:
            return np.zeros(0, dtype=np.float32), counts
        return np.concatenate(pairs).astype(np.float32), counts
