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

"""smlib's implementation of ocp_viewer_core's BackendProtocol: exact measurements for the
viewer's distance and properties tools.

The tools ask a measurement backend (`ocp_viewer_core.backend.ViewerBackend`), which holds
the geometry behind the mesh. Every host runs it in another process than the user's (VS
Code's `python -m ocp_vscode --backend`, Jupyter CadQuery's server extension, ocp-viewer's
server, build123d Studio's sidecar), so a shape reaches it as the bytes of `SmlibShow.serialize`
in the model message and is rebuilt there. Importing this module registers the class, which an
entry point of the group `ocp_viewer.kernels.backend` does in that process. The docs describe
the protocol: https://bernhard-42.github.io/ocp_viewer_docs/plugins/backend/

This class says what smlib knows about a shape. The core runs the sequence and makes the
panel's rows from it, as it does for OCCT shapes.
"""

from __future__ import annotations

import struct
from math import sqrt

import numpy as np
from ocp_tessellate.plugin import CURVE_TYPES, SURFACE_TYPES, Placement
from ocp_viewer_core.backend import BackendProtocol, ShapeProperties, register_plugin
from smlibpy import Edge, Face, Location, Part, ShapeList, Shell, Vector, Vertex, from_smp
from smlibpy.shapes import _Body

from ._wire import FORMAT, HEADER, KERNEL, POINTS
from .tessellation import EDGE_KIND_CODES, FACE_KIND_CODES
from .types import Point, SmlibShape

__all__ = ["KERNEL", "SmlibBackend"]

# the GeomAbs names of the codes the mesh carries (face_types, edge_types), so the
# backend's geom_type and the viewer's status line agree
_SURFACE_NAMES = {code: name for name, code in SURFACE_TYPES.items()}
_CURVE_NAMES = {code: name for name, code in CURVE_TYPES.items()}

# the faces whose center is their axis' origin (as measure.get_center takes it)
_AXIS_FACES = ("Sphere", "Torus", "SurfaceOfRevolution")
# the faces whose center is their center of mass projected onto their axis (measure._axis_center)
_AXIS_MIDDLE_FACES = ("Cylinder", "Cone")


class SmlibBackend(BackendProtocol[SmlibShape]):
    def deserialize(self, data: bytes) -> SmlibShape:
        """The shape of the bytes of `SmlibShow.serialize`: one item as itself, several as a
        ShapeList, as `from_smp` returns them."""
        if data[:1] != HEADER:
            raise ValueError(
                f"These bytes are in format {data[0]} of ocp_viewer_smlibpy, this version reads "
                f"format {FORMAT}: the plugin differs between the user's and the backend's environment"
            )
        how, rest = data[1:2], data[2:]
        if how != POINTS:
            return from_smp(rest)
        values = struct.unpack(f"<{len(rest) // 8}d", rest)
        vertices = [Vertex(*values[i : i + 3]) for i in range(0, len(values), 3)]
        return vertices[0] if len(vertices) == 1 else ShapeList(vertices)

    def moved(self, shape: SmlibShape, placement: Placement) -> SmlibShape:
        return shape.moved(Location(placement.matrix()))

    def parts(self, shape: SmlibShape) -> tuple[list[Face], list[Edge], list[Vertex]]:
        """Its faces, edges and vertices in smlib's order - the order of the tessellation's
        face and edge blocks, so index i here is the viewer's ``faces_i`` and ``edges_i``."""
        if isinstance(shape, Vertex):
            return [], [], [shape]
        if isinstance(shape, Edge):
            return [], [shape], list(shape.vertices())
        if isinstance(shape, ShapeList):
            if len(shape) > 0 and isinstance(shape[0], Vertex):
                return [], [], list(shape)
            return [], list(shape), list(shape.vertices())
        # seams included: the kernel's edge list the tessellation samples (and the viewer
        # numbers) has them, smlibpy's `edges()` leaves them out by default
        return list(shape.faces()), list(shape.edges(seams=True)), list(shape.vertices())

    def kind(self, shape: SmlibShape) -> tuple[str, str]:
        return self._shape_type(shape), self._geom_type(shape)

    def center(self, shape: SmlibShape) -> Point:
        """The center of a shape: of a circle its center, of a cylinder or cone face the axis
        point at its middle, of a sphere, torus or surface of revolution the origin of its
        axis, else the middle of an edge, the centroid of a face or the center of mass of a
        body (ocp_viewer_core.measure.get_center)."""
        if isinstance(shape, Vertex):
            return self._point(shape)
        if isinstance(shape, ShapeList) and isinstance(shape[0], Vertex):
            x, y, z = np.asarray([self._point(v) for v in shape]).mean(axis=0)
            return float(x), float(y), float(z)
        if isinstance(shape, Edge):
            if self._geom_type(shape) == "Circle":
                return self._point(shape.arc_center)
            return self._point(shape.center())
        if isinstance(shape, Face):
            if self._geom_type(shape) in _AXIS_MIDDLE_FACES:
                return self._axis_center(shape)
            if self._geom_type(shape) in _AXIS_FACES:
                return self._point(shape.axis.origin)
            return self._point(shape.center())
        return self._point(shape.center())

    def properties(self, shape: SmlibShape) -> ShapeProperties:
        """What smlib knows about a body, face, edge or vertex."""
        shape_type = self._shape_type(shape)
        geom_type = self._geom_type(shape)
        props: ShapeProperties = {"refpoint": None}

        if isinstance(shape, Vertex):
            props["point"] = props["refpoint"] = self._point(shape)

        elif isinstance(shape, ShapeList):
            props["refpoint"] = self.center(shape)

        elif isinstance(shape, Edge):
            if geom_type == "Circle":
                props["center"] = self._point(shape.arc_center)
                props["radius"] = shape.radius
                props["start"] = self._point(shape.point_at(0))
                if shape.is_closed:
                    props["refpoint"] = self._point(shape.point_at(0))
                else:
                    props["end"] = self._point(shape.point_at(1))
                    props["refpoint"] = self._point(shape.point_at(0.5))
            else:
                # a line, an ellipse, a B-spline, or another curve: its ends and its middle
                props["start"] = self._point(shape.point_at(0))
                props["middle"] = self._point(shape.point_at(0.5))
                props["end"] = self._point(shape.point_at(1))
                props["refpoint"] = self._point(shape.point_at(0.5))
            props["length"] = float(shape.length)
            props["tangent_start"] = self._point(shape.tangent_at(0))
            props["tangent_end"] = self._point(shape.tangent_at(1))

        elif isinstance(shape, Face):
            if geom_type == "Plane":
                # the center of mass, which a planar face contains (as OCCT's center has it)
                props["center"] = self._point(shape.center())
            elif geom_type == "Cylinder":
                props["center"] = self._axis_center(shape)
                props["radius"] = float(shape.radius)
            elif geom_type == "Cone":
                props["center"] = self._axis_center(shape)
                props["base_radius"] = float(shape.radius)
                props["half_angle"] = float(shape.cone_angle)
            elif geom_type == "Sphere":
                props["center"] = self._point(shape.axis.origin)
                props["radius"] = float(shape.radius)
            elif geom_type == "Torus":
                props["center"] = self._point(shape.axis.origin)
                props["minor_radius"] = float(shape.radius)
                props["major_radius"] = float(shape.major_radius)
            elif geom_type == "SurfaceOfRevolution":
                props["axis_location"] = self._point(shape.axis.origin)
                props["axis_direction"] = self._point(shape.axis.direction)
            props["refpoint"] = self._face_refpoint(shape)
            props["area"] = float(shape.area)
            props["normal"] = self._point(shape.normal_at())

        elif isinstance(shape, Part):
            props["volume"] = float(shape.volume)
            props["refpoint"] = self._point(shape.center())

        elif isinstance(shape, Shell):  # a sheet or a sketch
            props["area"] = float(shape.area)
            props["refpoint"] = self._point(shape.center())

        else:
            raise TypeError(f"not a shape the smlib backend measures: {type(shape).__name__}")

        if shape_type != "Vertex":
            bb = shape.bbox()
            props["bb_min"] = self._point(bb.min)
            props["bb_max"] = self._point(bb.max)

        return props

    def closest_points(self, shape1: SmlibShape, shape2: SmlibShape) -> tuple[float, Point, Point]:
        return self._closest_points(shape1, shape2)

    def closest_directions(self, shape1: SmlibShape, shape2: SmlibShape) -> tuple[Point, Point] | None:
        """The tangent of an edge or the normal of a face, at the closest points of the two."""
        if not isinstance(shape1, (Edge, Face)) or not isinstance(shape2, (Edge, Face)):
            return None
        _, p1, p2 = self._closest_points(shape1, shape2)
        return self._direction_at(shape1, p1), self._direction_at(shape2, p2)

    def _shape_type(self, shape: SmlibShape) -> str:
        if isinstance(shape, ShapeList):
            return "Compound"
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

    def _geom_type(self, shape: SmlibShape) -> str:
        """The GeomAbs name of the shape's geometry, as ocp_viewer_core.measure reports it
        for OCCT shapes and the mesh's face_types and edge_types encode it."""
        if isinstance(shape, Vertex):
            return "Point"
        if isinstance(shape, Edge):
            return _CURVE_NAMES[EDGE_KIND_CODES.get(shape.geom_type, CURVE_TYPES["OtherCurve"])]
        if isinstance(shape, Face):
            return _SURFACE_NAMES[FACE_KIND_CODES.get(shape.geom_type, SURFACE_TYPES["OtherSurface"])]
        return "Other"

    def _closest_points(self, shape1: SmlibShape, shape2: SmlibShape) -> tuple[float, Point, Point]:
        """The minimum distance of two shapes and the points where it is taken, one on each.

        A vertex is a point, and every other shape has the point nearest to a point. Two
        bodies, a body and a face or an edge, and two faces have smlibpy's `closest_points`
        (a face measured as a shell of its own). Two edges have no such method in smlibpy
        0.6.2; see `_edge_edge_closest`. A list of edges or vertices has the nearest of them.
        """
        if isinstance(shape1, ShapeList):
            return min((self._closest_points(item, shape2) for item in shape1), key=lambda found: found[0])
        if isinstance(shape2, ShapeList):
            return min((self._closest_points(shape1, item) for item in shape2), key=lambda found: found[0])
        if isinstance(shape1, Vertex):
            p1 = self._point(shape1)
            return self._distance(p1, self._nearest_point(shape2, p1))
        if isinstance(shape2, Vertex):
            p2 = self._point(shape2)
            return self._distance(self._nearest_point(shape1, p2), p2)

        if isinstance(shape1, Edge) and isinstance(shape2, Edge):
            return self._edge_edge_closest(shape1, shape2)

        # smlibpy measures from a body: a face as a shell of its own, and an edge from
        # the other side (swapping the points back)
        if isinstance(shape1, Edge):
            dist, p2, p1 = self._closest_points(shape2, shape1)
            return dist, p1, p2
        body1 = shape1.to_shell() if isinstance(shape1, Face) else shape1
        result = body1.closest_points(shape2)
        return float(result.distance), self._point(result.point), self._point(result.other_point)

    def _nearest_point(self, shape: _Body | Face | Edge | Vertex, point: Point) -> Point:
        if isinstance(shape, Vertex):
            return self._point(shape)
        return self._point(shape.closest_point(point))

    def _edge_edge_closest(
        self, edge1: Edge, edge2: Edge, samples: int = 64, seeds: int = 4
    ) -> tuple[float, Point, Point]:
        """The closest points of two edges by alternating projection: the points of
        `edge1` at `samples` fractions of its length give starting points, the nearest of
        `edge2` to each its distance; from the `seeds` best, the nearest point of either
        edge to the other's last point is taken in turn until the distance stops
        falling (the method smlibpy's own body distance starts with). Edges that cross
        end at the crossing, 0 apart."""
        fractions = np.linspace(0.0, 1.0, samples + 1)
        starts = []
        for t in fractions:
            p = self._point(edge1.point_at(float(t)))
            q = self._nearest_point(edge2, p)
            starts.append((self._norm(p, q), p, q))
        starts.sort(key=lambda s: s[0])

        tolerance = 1e-12 * max(float(edge1.length), float(edge2.length), 1.0)
        best: tuple[float, Point, Point] = starts[0]
        for dist, p, q in starts[:seeds]:
            for _ in range(200):
                p = self._nearest_point(edge1, q)
                q = self._nearest_point(edge2, p)
                previous, dist = dist, self._norm(p, q)
                if previous - dist < tolerance:
                    break
            if dist < best[0]:
                best = (dist, p, q)
        return best

    def _direction_at(self, shape: Edge | Face, point: Point) -> Point:
        """The tangent of an edge or the normal of a face at a point of it."""
        if isinstance(shape, Edge):
            return self._point(shape.tangent_at(shape.param_at_point(point)))
        return self._point(shape.normal_at(point))

    def _face_refpoint(self, face: Face) -> Point:
        """Where the Properties panel's guide line ends on a face: the surface point in the
        middle of the face's parameter box, as OCCT's center_of_geometry has it (for a cylinder
        the point opposite the seam, at half height). When the face does not cover that point -
        a trimmed face whose middle is cut away, such as a plate with a hole there - a point the
        kernel finds inside the face instead (a curved face's centroid can lie off it)."""
        middle = face.position_at(0.5, 0.5)
        if face.distance_to(middle) <= 1e-7 * max(face.bbox().diagonal, 1.0):
            return self._point(middle)
        return self._point(face.interior_point())

    def _axis_center(self, face: Face) -> Point:
        """The center of a cylinder or cone face: its center of mass projected onto its axis.

        The axis' origin is where the surface was built, anywhere on an infinite axis; this is
        the axis point at the face's middle, as ocp_viewer_core.measure._axis_center has it for
        OCCT faces."""
        origin = np.asarray(self._point(face.axis.origin))
        direction = np.asarray(self._point(face.axis.direction))
        com = np.asarray(self._point(face.center()))
        x, y, z = origin + float((com - origin) @ direction) * direction
        return float(x), float(y), float(z)

    def _point(self, value: Vector | Vertex | Point) -> Point:
        x, y, z = (float(c) for c in tuple(value))
        return (x, y, z)

    def _norm(self, p1: Point, p2: Point) -> float:
        dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        return sqrt(dx * dx + dy * dy + dz * dz)

    def _distance(self, p1: Point, p2: Point) -> tuple[float, Point, Point]:
        return self._norm(p1, p2), p1, p2


register_plugin(KERNEL, SmlibBackend(), api=1)
