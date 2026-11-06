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

"""OCCT objects for smlibpy objects, to measure them against OCCT objects. Not a part of a
protocol: a kernel plugin works without it."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

import numpy as np
from ocp_tessellate._kernel import binding as occt_binding, kernel
from smlibpy import Axis, Edge, Face, Location, Plane, Vector, Vertex, Wire, to_brep
from smlibpy.shapes import _Body

__all__ = ["to_occt"]


def to_occt(obj: object, binding: str | None = None) -> Any:
    """OCCT objects for smlibpy objects, so that ``show(to_occt(obj))`` measures
    like any OCCT object: parts, shells, sketches and faces through smlib's
    ``.brep`` writer, edges and wires as exact lines or NURBS curves, vertices
    and vectors as vertices, locations as ``TopLoc_Location``, planes as ``gp_Pln``, axes
    as ``gp_Ax1``; lists and dicts element-wise.

    ``binding`` is ``"OCP"`` or ``"nanocct"``; by default ocp_tessellate's
    ``kernel()``, nanocct where it is installed, else OCP. Raises ``ValueError``
    where smlib cannot write a body (offset surfaces of free-form faces): show
    those with the host's ``show``.
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
            return occ.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(pnt(e.start), pnt(e.end)).Edge()
        n = e.nurbs()
        uniq, mult = np.unique(np.round(n.knots, 12), return_counts=True)
        poles = array(pnt_array, n.points, pnt)
        knots = array(real_array, uniq, float)
        mults = array(int_array, mult, int)
        if n.weights is None:
            curve = occ.Geom.Geom_BSplineCurve(poles, knots, mults, n.degree)
        else:
            curve = occ.Geom.Geom_BSplineCurve(poles, array(real_array, n.weights, float), knots, mults, n.degree)
        return occ.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(curve).Edge()

    def location(loc: Location) -> Any:
        m = np.asarray(loc.matrix, dtype=np.float64)
        t = occ.gp.gp_Trsf()
        t.SetValues(*m[:3].ravel().tolist())
        return occ.TopLoc.TopLoc_Location(t)

    def body(b: _Body | Face) -> Any:
        try:
            text = to_brep(b)
        except Exception as ex:  # KernelError for offset surfaces, ValueError for empty bodies
            raise ValueError(
                f"to_occt: smlib cannot write this {type(b).__name__} as .brep ({ex}); "
                "show it with the host's show instead"
            ) from ex
        shape = occ.TopoDS.TopoDS_Shape()
        # OCP's Read_s reads a BytesIO, nanocct's a text stream
        stream = io.StringIO(text) if occ.__name__ == "nanocct" else io.BytesIO(text.encode("ascii"))
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
