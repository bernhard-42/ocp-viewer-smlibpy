"""TessellationProtocol and ShowProtocol.convert of the smlib plugin: the layout of the
Tessellation dict, the box against the OCCT path, the sampling of edges, the bounding
box, and convert."""

import numpy as np
import pytest
from ocp_tessellate._kernel import kernel
from ocp_tessellate.convert import tessellate_group, to_ocpgroup
from ocp_tessellate.ocp_utils import bounding_box as occt_bounding_box
from ocp_tessellate.plugin import CURVE_TYPES, SURFACE_TYPES, CoordAxis, CoordSystem
from ocp_tessellate.tessellator import compute_quality, tessellate as occt_tessellate
from smlibpy import (
    Axis,
    Box,
    Cone,
    Cylinder,
    Edge,
    Line,
    Plane,
    Pos,
    Rectangle,
    Sphere,
    ThreePointArc,
    Torus,
    Vector,
    Wire,
    Z,
)
from smlibpy.deformations import Lattice

from ocp_viewer_smlibpy.show import SmlibShow
from ocp_viewer_smlibpy.tessellation import SmlibTessellation

DEVIATION, ANGULAR = 0.1, 0.2


TESSELLATION = SmlibTessellation()
SHOW = SmlibShow()
bounding_box = TESSELLATION.bounding_box
cache_id = TESSELLATION.cache_id
discretize_edges = TESSELLATION.discretize_edges
convert = SHOW.convert


def tessellate(body, quality, angular_tolerance, compute_edges=True):
    return TESSELLATION.tessellate(body, 0.1, quality, angular_tolerance, compute_edges)


def quality_of(body):
    return compute_quality(bounding_box(body), deviation=DEVIATION)


def layout_invariants(t):
    v = t["vertices"].reshape(-1, 3)
    n = t["normals"].reshape(-1, 3)
    tri = t["triangles"].reshape(-1, 3)
    assert len(n) == len(v), "one normal per vertex"
    assert tri.min() >= 0 and tri.max() < len(v)
    assert t["triangles_per_face"].sum() == len(tri)
    assert t["segments_per_edge"].sum() * 6 == len(t["edges"])
    assert len(t["face_types"]) == len(t["triangles_per_face"])
    assert len(t["edge_types"]) == len(t["segments_per_edge"])
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)
    # the winding agrees with the normals
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    assert (np.einsum("ij,ij->i", fn, n[tri[:, 0]]) > 0).all()
    # every face owns a contiguous block of vertices, and the blocks are disjoint
    off, last_end = 0, -1
    for count in t["triangles_per_face"]:
        used = np.unique(tri[off : off + count])
        assert used.min() == last_end + 1 and used.max() - used.min() + 1 == len(used)
        last_end = used.max()
        off += count
    assert last_end == len(v) - 1


@pytest.mark.parametrize("body", [Box(1, 2, 3), Cylinder(1, 2), Sphere(1), Cone(1, 0, 2), Torus(3, 1),
                                  Box(1, 2, 3) + Cylinder(0.2, 4) + Pos(0, 0, 2) * Sphere(1)])
def test_the_layout_is_the_viewers(body):
    t = tessellate(body, quality_of(body), ANGULAR)
    layout_invariants(t)
    assert len(t["triangles_per_face"]) == len(body.faces())
    assert len(t["segments_per_edge"]) == len(body.edges(seams=True))
    assert len(t["obj_vertices"]) == 3 * len(body.vertices())


def test_the_unit_box_is_identical_to_the_occt_path():
    k = kernel()
    occt_box = k.BRepPrimAPI.BRepPrimAPI_MakeBox(1, 1, 1).Shape().Moved(k.TopLoc.TopLoc_Location(k.gp.gp_Trsf()))
    # OCCT's box has its corner at the origin; smlib's is centred
    trsf = k.gp.gp_Trsf()
    trsf.SetTranslation(k.gp.gp_Vec(-0.5, -0.5, -0.5))
    occt_box = occt_box.Moved(k.TopLoc.TopLoc_Location(trsf))
    quality = compute_quality(occt_bounding_box(occt_box), deviation=DEVIATION)
    theirs = occt_tessellate(occt_box, "unit-box", DEVIATION, quality, ANGULAR)
    mine = tessellate(Box(1, 1, 1), quality, ANGULAR)

    def pairs(t):
        v = t["vertices"].reshape(-1, 3)
        n = t["normals"].reshape(-1, 3)
        return sorted(map(tuple, np.round(np.hstack([v, n]), 5)))

    def segments(t):
        s = np.round(t["edges"].reshape(-1, 2, 3), 5)
        return sorted(tuple(np.sort(seg, axis=0).ravel()) for seg in s)

    assert pairs(mine) == pairs(theirs)
    assert segments(mine) == segments(theirs)
    assert sorted(mine["face_types"].tolist()) == sorted(theirs["face_types"].tolist())
    assert sorted(mine["edge_types"].tolist()) == sorted(theirs["edge_types"].tolist())


def test_the_apex_of_a_cone_is_split_per_normal():
    cone = Cone(1, 0, 2)
    t = tessellate(cone, quality_of(cone), ANGULAR)
    v = t["vertices"].reshape(-1, 3)
    n = t["normals"].reshape(-1, 3)
    apex = np.isclose(v[:, 2], v[:, 2].max())
    assert apex.sum() > 1 and len(np.unique(np.round(n[apex], 5), axis=0)) == apex.sum()


def test_face_and_edge_types_are_geomabs_codes():
    body = Box(1, 2, 3) + Cylinder(0.2, 4) + Pos(0, 0, 2) * Sphere(1)
    t = tessellate(body, quality_of(body), ANGULAR)
    assert set(t["face_types"].tolist()) == {SURFACE_TYPES["Plane"], SURFACE_TYPES["Cylinder"], SURFACE_TYPES["Sphere"]}
    assert set(t["edge_types"].tolist()) == {CURVE_TYPES["Line"], CURVE_TYPES["Circle"]}


def pillow():
    block = Box(60, 40, 20)
    lat = Lattice(block, 3)
    lat.move((1, 1, -1), (0, 0, 8))
    lat.scale(lat.layer(Axis.X, 0), (1, 0.8, 1))
    return lat.apply(block)


@pytest.mark.parametrize("body", [Cylinder(1, 2), Torus(3, 1), pillow()])
def test_edge_polylines_stay_within_a_tenth_of_the_quality(body):
    quality = quality_of(body)
    t = tessellate(body, quality, ANGULAR)
    segs = t["edges"].reshape(-1, 2, 3)
    start = 0
    for handle, count in zip(body._brep.edges(), t["segments_per_edge"]):
        s = segs[start : start + count]
        start += count
        t0, t1 = handle.parameter_range()
        c0, c1 = handle.curve().parameter_range()
        dense = np.array([handle.point_at(x) for x in np.linspace(max(t0, c0), min(t1, c1), 300)])
        a, b = s[:, 0].astype(float), s[:, 1].astype(float)
        d = dense[:, None, :] - a[None]
        ab = b - a
        l2 = (ab**2).sum(-1)
        u = np.clip((d * ab[None]).sum(-1) / np.where(l2 == 0, 1, l2), 0, 1)
        deviation = np.linalg.norm(d - u[..., None] * ab[None], axis=-1).min(axis=1).max()
        assert deviation <= quality / 10 * 1.01


def test_discretize_edges_of_a_rectangle_with_an_arc():
    wire = Wire([Line((0, 0, 0), (2, 0, 0)), Line((2, 0, 0), (2, 1, 0)),
                 ThreePointArc((2, 1, 0), (1, 1.5, 0), (0, 1, 0)), Line((0, 1, 0), (0, 0, 0))])
    deflection = compute_quality(bounding_box(wire.edges()), deviation=DEVIATION) / 10
    d = discretize_edges(wire.edges(), deflection)
    counts = d["segments_per_edge"].tolist()
    assert counts[0] == counts[1] == counts[3] == 1 and counts[2] > 10
    assert d["edge_types"].tolist() == [CURVE_TYPES["Line"], CURVE_TYPES["Line"], CURVE_TYPES["Circle"], CURVE_TYPES["Line"]]
    assert d["segments_per_edge"].sum() * 6 == len(d["edges"])
    corners = sorted(map(tuple, d["obj_vertices"].reshape(-1, 3).tolist()))
    assert corners == [(0, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    # the deflection decides (about 1.2e-4 here)
    assert discretize_edges(wire.edges(), 1e-5)["segments_per_edge"][2] > counts[2]
    assert discretize_edges(wire.edges(), 1e-3)["segments_per_edge"][2] < counts[2]


def test_a_flat_sketch_has_a_bounding_box_with_a_quality():
    bb = bounding_box(Rectangle(2, 1))
    assert bb.zsize == 0 and compute_quality(bb, deviation=DEVIATION) > 0


def converted(obj, name="b", color=None, alpha=None, helper_scale=1.0):
    return convert(obj, name, color, alpha, helper_scale)


def test_convert_picks_the_kind_and_the_building_block():
    box = Box(1, 2, 3)
    kinds = {name: converted(o, name).kind for name, o in (
        ("part", box), ("sketch", Rectangle(1, 1)), ("shell", box.shells()[0]), ("face", box.faces()[0]),
        ("edge", box.edges()[0]), ("wire", box.wires()[0]), ("vertex", box.vertices()[0]), ("vector", Vector(1, 1, 1)),
        ("location", Pos(1, 2, 3)), ("plane", Plane.XZ), ("axis", Z))}
    assert kinds == {"part": "solid", "sketch": "face", "shell": "shell", "face": "face", "edge": "edge", "wire": "edge",
                     "vertex": "vertex", "vector": "vertex", "location": "edge", "plane": "edge", "axis": "edge"}
    with pytest.raises(TypeError):
        converted(kernel().BRepPrimAPI.BRepPrimAPI_MakeBox(1, 1, 1).Shape(), "occt")


def test_the_helpers_are_the_kernels_lines():
    helper = converted(Pos(1, 2, 3), helper_scale=2.0)
    assert isinstance(helper, CoordSystem) and all(isinstance(e, Edge) for e in helper.objs)
    assert tuple(helper.objs[0].end) == pytest.approx((3, 2, 3))
    arrow = converted(Z, helper_scale=2.0)
    assert isinstance(arrow, CoordAxis) and len(arrow.objs) == 11 and all(isinstance(e, Edge) for e in arrow.objs)


def test_lists_and_dicts_are_not_converts_concern():
    box = Box(1, 2, 3)
    with pytest.raises(TypeError):
        converted([box], "list")
    with pytest.raises(TypeError):
        converted({"a": box}, "dict")


def test_convert_takes_a_bodys_own_color_unless_one_is_given():
    blue = Box(1, 1, 1).colored("steelblue")
    assert converted(blue).color.web_color == "#4682b4"
    assert converted(blue, color="red", alpha=0.5).color.web_color == "#ff0000"
    assert converted(Box(1, 1, 1)).color is None


def test_the_scene_goes_through_the_pipeline():
    box = Box(10, 10, 2)
    part_group, instances = to_ocpgroup(box, box.edges(), Pos(0, 0, 5), Z, names=["box", "edges", "loc", "axis"], helper_scale=3.0)
    meshed, shapes, mapping = tessellate_group(part_group, instances, {"deviation": DEVIATION, "angular_tolerance": ANGULAR})
    parts = {p["name"]: p for p in shapes["parts"]}
    assert parts["box"]["type"] == "shapes" and len(meshed) == 1
    assert "parts" in parts["edges"] and len(parts["edges"]["parts"]) == 12
    assert parts["loc"]["color"] == ["#ff0000", "#008000", "#0000ff"]
    seg = np.asarray(parts["axis"]["shape"]["edges"]).reshape(-1, 2, 3)
    assert abs(np.linalg.norm(seg[0, 1] - seg[0, 0]) - 3.0) < 1e-6
    # every leaf is in the backend's model, tagged for the smlib backend, the body, its
    # edges, the location and the axis, as OCCT's are
    entries = {m["id"]: m for m in mapping["parts"] if "shape" in m}
    assert set(entries) == {"/Group/box", "/Group/loc", "/Group/axis"}
    assert entries["/Group/box"]["shape"] == {"kernel": "smlib", "obj": box}
    assert entries["/Group/loc"]["shape"]["kernel"] == "smlib" and len(entries["/Group/loc"]["shape"]["obj"]) == 3
    assert entries["/Group/axis"]["shape"]["kernel"] == "smlib" and len(entries["/Group/axis"]["shape"]["obj"]) == 11
    edges = next(m for m in mapping["parts"] if m["id"] == "/Group/edges")
    assert len(edges["parts"]) == 12 and all(e["shape"]["kernel"] == "smlib" for e in edges["parts"])


def test_the_cache_id_is_the_geometry_not_the_object():
    """Equal for a body built twice, different for a moved copy, and what the pipeline
    shares a mesh under."""
    assert cache_id(Box(1, 2, 3)) == cache_id(Box(1, 2, 3))
    assert cache_id(Box(1, 2, 3)) != cache_id(Pos(1, 0, 0) * Box(1, 2, 3))
    box = Box(1, 2, 3)
    group, instances = to_ocpgroup(box, names=["b"])
    assert group.objects[0].cache_id == f"smlib:{cache_id(box)}" == instances[0]["cache_id"]

