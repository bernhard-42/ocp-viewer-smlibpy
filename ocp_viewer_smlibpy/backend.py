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

"""The backend contract of the smlib plugin: exact measurements for the viewer's tools.

The viewer's distance and properties tools ask a measurement backend
(`ocp_viewer_core.backend.ViewerBackend`), which holds the geometry that produced
the mesh and answers from it rather than from triangles. Every host runs the
backend in another process than the user's (VS Code's `python -m ocp_vscode
--backend`, Jupyter CadQuery's server extension, ocp-viewer's server, build123d
Studio's sidecar), so a body reaches it as bytes in the model message and is
rebuilt there (ocp-viewer-core's ExternalKernel.md describes the protocol). This module is the smlib half of that: what a body travels as, how
it is rebuilt, and the measurements - the same five functions a kernel plugin
registers with `ocp_viewer_core.backend.register_plugin`:

- `serialize(body) -> bytes`: the body as bytes, on the show side
- `load(data, loc) -> (body, faces, edges, vertices)`: the body and its sub-shapes
  rebuilt from the bytes and placed, on the backend side; the lists are in the order
  the tessellation (viewer.py) lists faces and edges (seams included), which is the
  order the viewer numbers them (`{id}/faces/faces_{i}`)
- `get_properties(shape)`: the properties of a body, face, edge or vertex
- `calc_distance(shape1, shape2, center)`: the minimum or center distance of two shapes
- `calc_angle(shape1, shape2)`: the angle of two edges or faces at their closest points

The responses have the layout of `ocp_viewer_core.measure`: the panel prints a
section's numbers as rows and its strings as they are, and `refpoint`,
`refpoint1`/`refpoint2` place the markers. The backend assembles the distance
tool's response from `calc_distance` and `calc_angle` with
`ocp_viewer_core.measure.distance_response`, as it does for OCCT shapes.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from ocp_tessellate.plugin import CURVE_TYPES, SURFACE_TYPES
from ocp_viewer_core.measure import with_supplement
from smlibpy import Edge, Face, Location, Part, Shell, Vector, Vertex, from_smb, to_smb
from smlibpy.shapes import _Body

from .viewer import EDGE_KIND_CODES, FACE_KIND_CODES, KERNEL, SmlibBody

__all__ = ["KERNEL", "calc_angle", "calc_distance", "get_properties", "load", "serialize"]

# a shape the backend measures: a body (the whole entry), a face, an edge or a vertex of it
SmlibShape = _Body | Face | Edge | Vertex
Point = tuple[float, float, float]
# a location on the wire: translation and quaternion (x, y, z, w) as ocp_tessellate's loc_to_tq
# writes them
WireLocation = tuple[tuple[float, float, float], tuple[float, float, float, float]]


# ===========================================================================
# The backend contract (ocp_viewer_core/plugins.py; __init__.py registers it)
#   1. serialize       the body as bytes, what it travels as (show side)
#   2. load            the body rebuilt from the bytes and placed, with its faces,
#                      edges and vertices in the order the viewer numbers them
#                      (backend side)
#   3. get_properties  the properties tool's response for one shape
#   4. calc_distance   the minimum or center distance of two shapes
#   5. calc_angle      the angle of two edges or faces at their closest points
# ===========================================================================


def serialize(body: SmlibBody) -> bytes:
    """The body as the bytes of an SMLib ``.smb`` file (binary, exact): what stands for
    it in the backend's model. A face is sent as a shell of its own, as it is drawn."""
    return to_smb(body, ascii=False)


def load(data: bytes, loc: WireLocation | None = None) -> tuple[SmlibBody, list[Face], list[Edge], list[Vertex]]:
    """The body of `serialize`'s bytes, moved by `loc` (translation and quaternion, as
    the model message carries a location), and its faces, edges and vertices in
    smlib's order - the order of the tessellation's face and edge blocks, so index i
    here is the viewer's ``faces_i`` and ``edges_i``."""
    body = from_smb(data)
    if loc is not None:
        body = body.moved(_location(loc))
    # seams included: the kernel's edge list the tessellation samples (and the viewer
    # numbers) has them, smlibpy's `edges()` leaves them out by default
    return body, list(body.faces()), list(body.edges(seams=True)), list(body.vertices())


def get_properties(shape: SmlibShape) -> dict[str, Any]:
    """The properties of a body, face, edge or vertex: the sections the panel shows,
    `shape_type`, `geom_type` and the `refpoint` of the marker - the layout of
    ocp_viewer_core.measure.get_properties."""
    shape_type = _shape_type(shape)
    geom_type = _geom_type(shape)

    refpoint: Point | None = None
    sections: list[dict[str, Any]] = []

    if isinstance(shape, Vertex):
        pt = _point(shape)
        refpoint = pt
        sections.append({"xyz": pt})

    elif isinstance(shape, Edge):
        geom_section: dict[str, Any] = {}
        pos_section: dict[str, Any] = {}

        if geom_type == "Circle":
            radius = shape.radius
            geom_section["center"] = _point(shape.arc_center)
            geom_section["radius / diam"] = f"{radius:.3f} /  {2 * radius:.3f}"
            pos_section["start"] = _point(shape.point_at(0))
            if shape.is_closed:
                refpoint = _point(shape.point_at(0))
            else:
                pos_section["end"] = _point(shape.point_at(1))
                refpoint = _point(shape.point_at(0.5))

        else:
            # a line, an ellipse, a B-spline, or another curve: its ends and its middle
            pos_section["start"] = _point(shape.point_at(0))
            pos_section["middle"] = _point(shape.point_at(0.5))
            pos_section["end"] = _point(shape.point_at(1))
            refpoint = _point(shape.point_at(0.5))

        if len(geom_section) > 0:
            sections.append(geom_section)
        sections.append(pos_section)

        meas_section: dict[str, Any] = {"length": float(shape.length)}
        for i in (0, 1):
            angle = abs(90 - _angle_between(shape.tangent_at(i), (0, 0, 1)))
            meas_section[f"angle@{i} to XY"] = with_supplement(angle)
        sections.append(meas_section)

    elif isinstance(shape, Face):
        geom_section = {}

        if geom_type == "Plane":
            geom_section["center"] = _point(shape.plane.origin)

        elif geom_type == "Cylinder":
            geom_section["center"] = _point(shape.axis.origin)
            geom_section["radius"] = float(shape.radius)

        elif geom_type == "Cone":
            geom_section["center"] = _point(shape.axis.origin)
            geom_section["base radius"] = float(shape.radius)
            geom_section["half angle"] = float(shape.cone_angle)

        elif geom_type == "Sphere":
            geom_section["center"] = _point(shape.axis.origin)
            geom_section["radius"] = float(shape.radius)

        elif geom_type == "Torus":
            geom_section["center"] = _point(shape.axis.origin)
            geom_section["minor radius"] = float(shape.radius)
            geom_section["major radius"] = float(shape.major_radius)

        elif geom_type == "SurfaceOfRevolution":
            geom_section["axe loc"] = _point(shape.axis.origin)
            geom_section["axe dir"] = _point(shape.axis.direction)
            geom_type = "Revolution"

        if len(geom_section) > 0:
            sections.append(geom_section)

        # a point on the face, where the marker goes (a curved face's centroid can lie off it)
        refpoint = _point(shape.interior_point())

        meas_section = {"area": float(shape.area)}
        angle = _angle_between(shape.normal_at(), (0, 0, 1))
        meas_section["angle to XY"] = with_supplement(angle)
        sections.append(meas_section)

    elif isinstance(shape, Part):
        sections.append({"volume": float(shape.volume)})
        refpoint = _point(shape.center())

    elif isinstance(shape, Shell):  # a sheet or a sketch
        sections.append({"area": float(shape.area)})
        refpoint = _point(shape.center())

    else:
        raise TypeError(f"not a shape the smlib backend measures: {type(shape).__name__}")

    if shape_type != "Vertex":
        bb = shape.bbox()
        sections.append({
            "bb": {
                "min": _point(bb.min),
                "center": _point(bb.center),
                "max": _point(bb.max),
                "size": _point(bb.size),
            }
        })

    return {
        "shape_type": shape_type,
        "geom_type": geom_type,
        "refpoint": refpoint,
        "result": sections,
    }


def calc_distance(shape1: SmlibShape, shape2: SmlibShape, center: bool = False) -> dict[str, Any]:
    """The distance of two shapes: between their centers (`center`), else the minimum,
    with the points it is taken between - the layout of ocp_viewer_core.measure.calc_distance."""
    if center:
        p1 = _center(shape1)
        p2 = _center(shape2)
        dist = _norm(p1, p2)
    else:
        dist, p1, p2 = _closest_points(shape1, shape2)
    return {
        "distance": dist,
        "⇒ X | Y | Z": [abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), abs(p2[2] - p1[2])],
        "point 1": p1,
        "point 2": p2,
        "info": "center" if center else "min",
        "refpoint1": p1,
        "refpoint2": p2,
    }


def calc_angle(shape1: SmlibShape, shape2: SmlibShape) -> dict[str, Any] | None:
    """The angle between two edges or faces at their closest points: of the tangents of
    edges, the normals of faces, and of an edge to a face; None for other shapes - the
    layout of ocp_viewer_core.measure.calc_angle."""
    is_edge1 = isinstance(shape1, Edge)
    is_edge2 = isinstance(shape2, Edge)
    if not (is_edge1 or isinstance(shape1, Face)) or not (is_edge2 or isinstance(shape2, Face)):
        return None

    _, p1, p2 = _closest_points(shape1, shape2)
    dir1 = _direction_at(shape1, p1)
    dir2 = _direction_at(shape2, p2)

    angle = _angle_between(dir1, dir2)
    # exactly one edge (a tangent against a normal): the angle of the edge to the surface
    if is_edge1 != is_edge2:
        angle = abs(90 - angle)

    response: dict[str, Any] = {"angle": angle}
    response["reference 1"] = _direction_label(shape1, 1)
    response["direction 1" if is_edge1 else "normal 1"] = dir1
    response["reference 2"] = _direction_label(shape2, 2)
    response["direction 2" if is_edge2 else "normal 2"] = dir2
    return response


# ---------------------------------------------------------------------------
# Private helpers for the backend contract
# ---------------------------------------------------------------------------


# the GeomAbs names of the codes the mesh carries (face_types, edge_types), so the
# backend's geom_type and the viewer's status line agree
_SURFACE_NAMES = {code: name for name, code in SURFACE_TYPES.items()}
_CURVE_NAMES = {code: name for name, code in CURVE_TYPES.items()}

# the faces whose center is their axis' origin (as measure.get_center takes it)
_AXIS_FACES = ("Cylinder", "Cone", "Sphere", "Torus", "SurfaceOfRevolution")


def _location(loc: WireLocation) -> Location:
    """A location from a translation and a quaternion (x, y, z, w): the rigid motion
    P' = R P + t that OCCT's gp_Trsf applies, as a 4x4 matrix."""
    (tx, ty, tz), (x, y, z, w) = loc
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = [tx, ty, tz]
    return Location(matrix)


def _shape_type(shape: SmlibShape) -> str:
    if isinstance(shape, Vertex):
        return "Vertex"
    if isinstance(shape, Edge):
        return "Edge"
    if isinstance(shape, Face):
        return "Face"
    if isinstance(shape, Part):
        return "Solid"
    if isinstance(shape, Shell):
        return "Shell"
    raise TypeError(f"not a shape the smlib backend measures: {type(shape).__name__}")


def _geom_type(shape: SmlibShape) -> str:
    """The GeomAbs name of the shape's geometry, as ocp_viewer_core.measure reports it
    for OCCT shapes and the mesh's face_types and edge_types encode it."""
    if isinstance(shape, Vertex):
        return "Point"
    if isinstance(shape, Edge):
        return _CURVE_NAMES[EDGE_KIND_CODES.get(shape.geom_type, CURVE_TYPES["OtherCurve"])]
    if isinstance(shape, Face):
        return _SURFACE_NAMES[FACE_KIND_CODES.get(shape.geom_type, SURFACE_TYPES["OtherSurface"])]
    return "Other"


def _center(shape: SmlibShape) -> Point:
    """The center of a shape: of a circle its center, of a cylinder, cone, sphere, torus
    or surface of revolution the origin of its axis, else the middle of an edge, the
    centroid of a face or the center of mass of a body (ocp_viewer_core.measure.get_center)."""
    if isinstance(shape, Vertex):
        return _point(shape)
    if isinstance(shape, Edge):
        if _geom_type(shape) == "Circle":
            return _point(shape.arc_center)
        return _point(shape.center())
    if isinstance(shape, Face):
        if _geom_type(shape) in _AXIS_FACES:
            return _point(shape.axis.origin)
        return _point(shape.center())
    return _point(shape.center())


def _closest_points(shape1: SmlibShape, shape2: SmlibShape) -> tuple[float, Point, Point]:
    """The minimum distance of two shapes and the points where it is taken, one on each.

    A vertex is a point, and every other shape has the point nearest to a point. Two
    bodies, a body and a face or an edge, and two faces have smlibpy's `closest_points`
    (a face measured as a shell of its own). Two edges have no such method in smlibpy
    0.6.2; see `_edge_edge_closest`.
    """
    if isinstance(shape1, Vertex):
        p1 = _point(shape1)
        return _distance(p1, _nearest_point(shape2, p1))
    if isinstance(shape2, Vertex):
        p2 = _point(shape2)
        return _distance(_nearest_point(shape1, p2), p2)

    if isinstance(shape1, Edge) and isinstance(shape2, Edge):
        return _edge_edge_closest(shape1, shape2)

    # smlibpy measures from a body: a face as a shell of its own, and an edge from
    # the other side (swapping the points back)
    if isinstance(shape1, Edge):
        dist, p2, p1 = _closest_points(shape2, shape1)
        return dist, p1, p2
    body1 = shape1.to_shell() if isinstance(shape1, Face) else shape1
    result = body1.closest_points(shape2)
    return float(result.distance), _point(result.point), _point(result.other_point)


def _nearest_point(shape: SmlibShape, point: Point) -> Point:
    if isinstance(shape, Vertex):
        return _point(shape)
    return _point(shape.closest_point(point))


def _edge_edge_closest(edge1: Edge, edge2: Edge, samples: int = 64, seeds: int = 4) -> tuple[float, Point, Point]:
    """The closest points of two edges by alternating projection: the points of
    `edge1` at `samples` fractions of its length give starting points, the nearest of
    `edge2` to each its distance; from the `seeds` best, the nearest point of either
    edge to the other's last point is taken in turn until the distance stops
    falling (the method smlibpy's own body distance starts with). Edges that cross
    end at the crossing, 0 apart."""
    fractions = np.linspace(0.0, 1.0, samples + 1)
    starts = []
    for t in fractions:
        p = _point(edge1.point_at(float(t)))
        q = _nearest_point(edge2, p)
        starts.append((_norm(p, q), p, q))
    starts.sort(key=lambda s: s[0])

    tolerance = 1e-12 * max(float(edge1.length), float(edge2.length), 1.0)
    best: tuple[float, Point, Point] = starts[0]
    for dist, p, q in starts[:seeds]:
        for _ in range(200):
            p = _nearest_point(edge1, q)
            q = _nearest_point(edge2, p)
            previous, dist = dist, _norm(p, q)
            if previous - dist < tolerance:
                break
        if dist < best[0]:
            best = (dist, p, q)
    return best


def _direction_label(shape: Edge | Face, shape_index: int) -> str:
    """What the direction of a shape in calc_angle is: a tangent or a normal."""
    if isinstance(shape, Edge):
        if _geom_type(shape) == "Line":
            return "line"
        return f"tangent at P{shape_index}"
    if _geom_type(shape) == "Plane":
        return "face normal"
    return f"surface normal at P{shape_index}"


def _direction_at(shape: Edge | Face, point: Point) -> Point:
    """The tangent of an edge or the normal of a face at a point of it."""
    if isinstance(shape, Edge):
        return _point(shape.tangent_at(shape.param_at_point(point)))
    return _point(shape.normal_at(point))


def _point(value: Vector | Vertex | Point) -> Point:
    x, y, z = (float(c) for c in tuple(value))
    return (x, y, z)


def _norm(p1: Point, p2: Point) -> float:
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _distance(p1: Point, p2: Point) -> tuple[float, Point, Point]:
    return _norm(p1, p2), p1, p2


def _angle_between(direction1: Vector | Point, direction2: Vector | Point) -> float:
    """The angle of two directions in degrees, 0 to 180."""
    a = np.asarray(tuple(direction1), dtype=float)
    b = np.asarray(tuple(direction2), dtype=float)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
