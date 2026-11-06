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

"""smlib's implementation of ocp_tessellate's ShowProtocol: what makes the host's `show`
and `show_all` take smlibpy's objects. Importing this module registers them, which an entry
point of the group `ocp_viewer.kernels.show` does in the user's process."""

from __future__ import annotations

import struct
from collections.abc import Sequence

from ocp_tessellate.plugin import CoordAxis, CoordSystem, OcpObject, ShowProtocol, object_color, register_show
from ocp_tessellate.types import ColorLike
from smlibpy import Axis, Edge, Face, Line, Location, Part, Plane, Shell, Sketch, Vector, Vertex, Wire, to_smp
from smlibpy.shapes import _Body

from ._wire import HEADER, KERNEL, POINTS, SMP
from .tessellation import SmlibTessellation
from .types import Point, SmlibObject

__all__ = ["SmlibShow"]


class SmlibShow(ShowProtocol[SmlibObject]):
    def is_object(self, obj: object) -> bool:
        """A body, face, edge, wire, vertex, vector, location, plane or axis. Lists and
        dicts are the host's, smlibpy's ShapeList too."""
        return isinstance(obj, SmlibObject)

    def line(self, start: Point, end: Point) -> Edge:
        return Line(start, end)

    def serialize(self, shape: SmlibObject | Sequence[SmlibObject]) -> bytes:
        """A body, face, edge, wire or a list of them as the bytes of an SMLib ``.smp`` file
        (binary, exact); vertices and vectors as their coordinates, which ``.smp`` does not
        hold. A face is sent as a shell of its own, as it is drawn. The first byte is the
        format's version: the backend's process can have another version of this plugin."""
        items = shape if isinstance(shape, (list, tuple)) else [shape]
        points = [item for item in items if isinstance(item, (Vertex, Vector))]
        if len(points) > 0 and len(points) == len(items):
            return HEADER + POINTS + struct.pack(f"<{3 * len(points)}d", *(c for point in points for c in tuple(point)))
        return HEADER + SMP + to_smp(shape, ascii=False)

    def convert(
        self, obj: SmlibObject, name: str, color: ColorLike | None, alpha: float | None, helper_scale: float
    ) -> OcpObject | CoordSystem | CoordAxis:
        """A shape as an OcpObject holding it, a location or plane as a CoordSystem, an axis
        as a CoordAxis. Without a color, a shape's own `color` is used, else the pipeline
        fills in the host's default of the kind."""
        if isinstance(obj, Location):
            return CoordSystem(
                name, tuple(obj.position), tuple(obj.x_dir), tuple(obj.z_dir), size=helper_scale, like=obj
            )
        if isinstance(obj, Plane):
            return CoordSystem(name, tuple(obj.origin), tuple(obj.x_dir), tuple(obj.z_dir), size=helper_scale, like=obj)
        if isinstance(obj, Axis):
            return CoordAxis(name, tuple(obj.origin), tuple(obj.direction), color, size=helper_scale, like=obj)
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
                raise TypeError(f"ocp_viewer_smlibpy: cannot show a {type(obj).__name__}")
            return OcpObject(kind, body, name=name, color=object_color(body, color, alpha))
        if isinstance(obj, (Edge, Wire)):
            edges = list(obj.edges()) if isinstance(obj, Wire) else [obj]
            return OcpObject("edge", edges, name=name, color=object_color(obj, color, alpha))
        if isinstance(obj, (Vertex, Vector)):
            return OcpObject("vertex", [obj], name=name, color=object_color(obj, color, alpha))
        raise TypeError(f"ocp_viewer_smlibpy: cannot show a {type(obj).__name__}")


register_show(KERNEL, SmlibTessellation(), SmlibShow(), api=1)
