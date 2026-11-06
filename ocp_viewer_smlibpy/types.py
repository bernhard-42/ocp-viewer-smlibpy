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

"""The types the plugin's modules name. Importing this registers nothing, so either half of the
plugin can import it without the other."""

from __future__ import annotations

from typing import Any

from smlibpy import Axis, Edge, Face, Location, Plane, ShapeList, Vector, Vertex, Wire
from smlibpy.shapes import _Body

# an object the host's `show` takes: a body, a sketch, a face, an edge, a wire, a vertex, a
# vector, a location, a plane or an axis
SmlibObject = _Body | Face | Edge | Wire | Vertex | Vector | Location | Plane | Axis

# a shape the backend measures: a body (the whole entry), a face, an edge or a vertex of it,
# or a list of edges or vertices, the whole entry of a helper or of a group of them
SmlibShape = _Body | Face | Edge | Vertex | ShapeList

Point = tuple[float, float, float]

# a kernel handle of smlibpy's _core (Brep, Edge, Curve): opaque here
KernelHandle = Any
