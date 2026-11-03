"""to_occt: smlibpy objects as OCCT shapes, in every installed binding."""

import importlib.util

import pytest
from ocp_tessellate._kernel import binding_of
from ocp_tessellate.ocp_utils import length, volume
from smlibpy import Box, Circle, Cylinder, Plane, Pos, Rectangle, Spline, Vector, Z, loft

from ocp_viewer_smlibpy import to_occt

BINDINGS = [b for b in ("OCP", "nanocct") if importlib.util.find_spec(b) is not None]


@pytest.mark.parametrize("binding", BINDINGS)
def test_bodies_edges_and_points_convert_exactly(binding):
    box = Box(1, 2, 3)
    circle = Cylinder(1, 2).edges().circles()[0]
    spline = Spline((0, 0, 0), (1, 1, 0), (2, 0, 1))
    out = to_occt({"box": box, "sketch": Rectangle(2, 1), "circle": circle, "spline": spline, "wire": box.wires()[0],
                   "vertex": box.vertices()[0], "vector": Vector(1, 2, 3), "plane": Plane.XZ, "loc": Pos(1, 2, 3), "axis": Z},
                  binding=binding)
    assert {binding_of(o).__name__ for o in out.values()} == {binding}
    assert abs(volume(out["box"]) - 6) < 1e-9
    assert abs(length(out["circle"]) - circle.length) < 1e-6
    assert abs(length(out["spline"]) - spline.length) < 1e-6
    assert abs(length(out["wire"]) - 6) < 1e-9
    assert type(out["plane"]).__name__ == "gp_Pln" and type(out["axis"]).__name__ == "gp_Ax1"
    assert type(out["loc"]).__name__ == "TopLoc_Location"


def test_an_offset_surface_is_refused_with_a_hint():
    part = loft([Rectangle(2, 2), Pos(0, 0, 3) * Circle(0.5)])
    free_form = part.faces().filter_by(lambda f: f.geom_type == "bspline")[0]
    with pytest.raises(ValueError, match="smlib_show"):
        to_occt(free_form.offset(0.2))
