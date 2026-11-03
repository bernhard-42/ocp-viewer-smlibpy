"""The backend contract of the smlib plugin (ocp_viewer_smlibpy.backend): the bytes a
body travels as, the ids the loaded body answers to, the properties and distances
against ocp_viewer_core's OCCT measurements of the same geometry, and the whole way
through ocp_viewer_core's ViewerBackend, registered as the package import does it."""

import base64

import numpy as np
import orjson
import pytest
from ocp_tessellate.convert import tessellate_group, to_ocpgroup
from ocp_tessellate.ocp_utils import tq_to_loc
from ocp_tessellate.plugin import external_scene
from ocp_tessellate.tessellator import compute_quality, get_edges, get_faces, get_vertices
from ocp_viewer_core import measure as occt
from ocp_viewer_core.backend import ViewerBackend
from ocp_viewer_core.codec import default
from ocp_viewer_core.comms import MessageType
from ocp_viewer_core.measure import distance_response
from ocp_viewer_core.plugins import PLUGINS, encode_kernel_shapes, plugin_of
from smlibpy import Box, Cylinder, Pos, Rectangle, Rot, Sphere, Torus, Vector, Vertex

from ocp_viewer_smlibpy import backend, to_occt
from ocp_viewer_smlibpy.backend import (
    KERNEL,
    calc_angle,
    calc_distance,
    get_properties,
    load,
    serialize,
)
from ocp_viewer_smlibpy.viewer import bounding_box, convert, tessellate

DEVIATION, ANGULAR = 0.1, 0.2


def test_importing_the_package_registered_the_backend():
    assert KERNEL in PLUGINS
    plugin = plugin_of(KERNEL)
    assert plugin.load is load and plugin.serialize is serialize
    assert plugin.get_properties is get_properties
    assert plugin.calc_distance is calc_distance and plugin.calc_angle is calc_angle


def plate():
    """A box with a hole: planes, a cylinder, lines, circles and a seam."""
    return Box(20, 30, 10) - Cylinder(4, 10).moved(Pos(5, 5, 0))


def entry(id_, body, loc=None):
    """A mapping entry as OcpExternalObject.collect emits it for a kernel body."""
    return {"id": id_, "shape": {"kernel": KERNEL, "obj": body}, "loc": loc}


def wire(mapping):
    """The mapping as the backend receives it: encoded, then through JSON."""
    return orjson.loads(orjson.dumps(encode_kernel_shapes(mapping)))


def section(response, key):
    for s in response["result"]:
        if key in s:
            return s[key]
    raise KeyError(key)


def _key(properties):
    """What identifies a shape across the kernels: its geometry's type and its box (a
    vertex: its point)."""
    if properties["shape_type"] == "Vertex":
        return ("Vertex", tuple(np.round(properties["refpoint"], 3)))
    bb = section(properties, "bb")
    return (properties["geom_type"], tuple(np.round(bb["center"], 3)), tuple(np.round(bb["size"], 3)))


class Twin:
    """A body loaded from its bytes beside its OCCT conversion, with the OCCT sub-shape
    of every smlib sub-shape: smlib and OCCT enumerate faces and edges in different
    orders, and the two kernels put the seam of a cylinder in different places, so seam
    edges have no twin."""

    def __init__(self, body):
        self.body, self.faces, self.edges, self.vertices = load(serialize(body))
        self.occt = to_occt(body)
        self._twins = {}
        for shapes in (get_faces(self.occt), get_edges(self.occt), get_vertices(self.occt)):
            for shape in shapes:
                self._twins[_key(occt.get_properties(shape))] = shape

    def twin(self, shape):
        return self._twins[_key(get_properties(shape))]

    def face(self, geom_type="plane", index=0):
        return [f for f in self.faces if f.geom_type == geom_type][index]

    def edge(self, geom_type="line", index=0):
        return [e for e in self.edges if e.geom_type == geom_type and not e.is_seam][index]


# ---------------------------------------------------------------------------
# serialize and load
# ---------------------------------------------------------------------------


def test_the_bytes_are_an_smb_file_and_load_as_the_same_body():
    body = plate()
    data = serialize(body)
    assert isinstance(data, bytes) and len(data) > 0
    loaded, faces, edges, vertices = load(data)
    assert loaded.volume == pytest.approx(body.volume)
    assert len(faces) == len(body.faces())
    assert len(edges) == len(body.edges(seams=True)) == len(body.edges()) + 1
    assert len(vertices) == len(body.vertices())


def test_a_loaded_body_answers_to_the_ids_the_tessellation_numbers():
    """Index i of the loaded faces, edges and vertices is face/edge/vertex block i of the
    mesh: every triangle of block i lies on loaded face i, every segment point of edge
    block i on loaded edge i (the seam too), and the vertices are the same points."""
    body = plate()
    mesh = tessellate(body, compute_quality(bounding_box(body), deviation=DEVIATION), ANGULAR)
    _, faces, edges, vertices = load(serialize(body))
    assert len(faces) == len(mesh["triangles_per_face"])
    assert len(edges) == len(mesh["segments_per_edge"])

    vertices_xyz = mesh["vertices"].reshape(-1, 3)
    triangles = mesh["triangles"].reshape(-1, 3)
    start = 0
    for i, count in enumerate(mesh["triangles_per_face"]):
        block = triangles[start : start + count]
        start += count
        centroids = vertices_xyz[block].mean(axis=1)
        for centroid in centroids[:: max(1, len(centroids) // 5)]:
            assert faces[i].distance_to(tuple(map(float, centroid))) < DEVIATION
        assert backend._geom_type(faces[i]) == backend._SURFACE_NAMES[int(mesh["face_types"][i])]

    points = mesh["edges"].reshape(-1, 2, 3)
    start = 0
    for i, count in enumerate(mesh["segments_per_edge"]):
        block = points[start : start + count]
        start += count
        for p in block[:, 0, :][:: max(1, count // 4)]:
            assert edges[i].distance_to(tuple(map(float, p))) < DEVIATION / 10 + 1e-6
        assert backend._geom_type(edges[i]) == backend._CURVE_NAMES[int(mesh["edge_types"][i])]

    assert np.allclose(mesh["obj_vertices"].reshape(-1, 3), [tuple(v) for v in vertices], atol=1e-6)


def test_a_location_on_the_wire_moves_the_body_as_occt_moves_it():
    body = Box(4, 6, 2)
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    half = 0.35
    t, q = (1.0, 2.0, 3.0), (*(axis * np.sin(half)), float(np.cos(half)))
    loaded, _, _, _ = load(serialize(body), (t, q))

    moved_occt = to_occt(body).Moved(tq_to_loc(t, q))
    bb_occt = section(occt.get_properties(moved_occt), "bb")
    bb = section(get_properties(loaded), "bb")
    assert np.allclose(bb["min"], bb_occt["min"], atol=1e-6) and np.allclose(bb["max"], bb_occt["max"], atol=1e-6)


def test_a_body_measured_before_it_is_serialized_still_loads_in_order():
    body = plate()
    before = [backend._center(f) for f in body.faces()]
    _ = body.volume, body.bbox(), [f.center() for f in body.faces()]
    _, faces, _, _ = load(serialize(body))
    assert np.allclose([backend._center(f) for f in faces], before, atol=1e-9)


# ---------------------------------------------------------------------------
# properties against the OCCT measurements of the same geometry
# ---------------------------------------------------------------------------


def test_face_properties_match_occt():
    twin = Twin(plate())
    for face in twin.faces:
        ours, theirs = get_properties(face), occt.get_properties(twin.twin(face))
        assert ours["shape_type"] == theirs["shape_type"] == "Face"
        assert ours["geom_type"] == theirs["geom_type"]
        assert section(ours, "area") == pytest.approx(section(theirs, "area"), rel=1e-6)
        assert section(ours, "angle to XY") == section(theirs, "angle to XY")
        assert np.allclose(section(ours, "bb")["min"], section(theirs, "bb")["min"], atol=1e-6)
        if ours["geom_type"] == "Cylinder":
            assert section(ours, "radius") == pytest.approx(section(theirs, "radius"))
            assert np.allclose(section(ours, "center")[:2], section(theirs, "center")[:2])
        assert face.distance_to(ours["refpoint"]) < 1e-9


def test_edge_properties_match_occt():
    twin = Twin(plate())
    for edge in twin.edges:
        if edge.is_seam:
            continue
        ours, theirs = get_properties(edge), occt.get_properties(twin.twin(edge))
        assert ours["shape_type"] == "Edge"
        assert ours["geom_type"] == theirs["geom_type"]
        assert section(ours, "length") == pytest.approx(section(theirs, "length"), rel=1e-9)
        assert section(ours, "angle@0 to XY") == section(theirs, "angle@0 to XY")
        if ours["geom_type"] == "Circle":
            assert section(ours, "radius / diam") == section(theirs, "radius / diam")
            assert np.allclose(section(ours, "center"), section(theirs, "center"), atol=1e-9)
        else:
            assert np.allclose(section(ours, "middle"), section(theirs, "middle"), atol=1e-9)
        assert edge.distance_to(ours["refpoint"]) < 1e-9


def test_a_seam_edge_has_properties_too():
    twin = Twin(plate())
    seam = next(e for e in twin.edges if e.is_seam)
    response = get_properties(seam)
    assert response["geom_type"] == "Line" and section(response, "length") == pytest.approx(10)


def test_vertex_properties():
    twin = Twin(plate())
    vertex = twin.vertices[0]
    ours, theirs = get_properties(vertex), occt.get_properties(twin.twin(vertex))
    assert ours["shape_type"] == "Vertex" and ours["geom_type"] == "Point"
    assert ours["result"] == [{"xyz": tuple(vertex)}]
    assert np.allclose(ours["refpoint"], theirs["refpoint"], atol=1e-9)


def test_body_properties_match_occt():
    body = plate()
    loaded, _, _, _ = load(serialize(body))
    ours, theirs = get_properties(loaded), occt.get_properties(to_occt(body))
    assert ours["shape_type"] == "Solid"
    assert section(ours, "volume") == pytest.approx(section(theirs, "volume"), rel=1e-6)
    assert np.allclose(ours["refpoint"], theirs["refpoint"], atol=1e-6)
    assert np.allclose(section(ours, "bb")["min"], section(theirs, "bb")["min"], atol=1e-6)

    sketch = Rectangle(4, 2)
    loaded, _, _, _ = load(serialize(sketch))
    response = get_properties(loaded)
    assert response["shape_type"] == "Shell" and section(response, "area") == pytest.approx(8)


def test_sphere_and_torus_faces_report_their_geometry():
    _, faces, _, _ = load(serialize(Sphere(3).moved(Pos(1, 2, 3))))
    sphere = get_properties(faces[0])
    assert sphere["geom_type"] == "Sphere" and section(sphere, "radius") == pytest.approx(3)
    assert np.allclose(section(sphere, "center"), (1, 2, 3))

    _, faces, _, _ = load(serialize(Torus(5, 1)))
    torus = get_properties(faces[0])
    assert torus["geom_type"] == "Torus"
    assert section(torus, "major radius") == pytest.approx(5) and section(torus, "minor radius") == pytest.approx(1)


# ---------------------------------------------------------------------------
# distances and angles against OCCT
# ---------------------------------------------------------------------------


def twins():
    return Twin(plate()), Twin(Box(5, 5, 5).moved(Pos(40, 0, 0) * Rot(z=30)))


def pairs():
    """Pairs of sub-shapes to measure between, as picks from the two twins."""
    return [
        ("face-face", lambda a: a.face(), lambda b: b.face()),
        ("cylinder-face", lambda a: a.face("cylinder"), lambda b: b.face(index=2)),
        ("edge-edge", lambda a: a.edge(), lambda b: b.edge()),
        ("skew-edges", lambda a: a.edge(index=1), lambda b: b.edge(index=3)),
        ("circle-edge", lambda a: a.edge("circle"), lambda b: b.edge(index=3)),
        ("vertex-face", lambda a: a.vertices[0], lambda b: b.face(index=1)),
        ("vertex-edge", lambda a: a.vertices[2], lambda b: b.edge(index=1)),
        ("vertex-vertex", lambda a: a.vertices[0], lambda b: b.vertices[0]),
        ("edge-face", lambda a: a.edge(index=1), lambda b: b.face()),
        ("body-body", lambda a: a.body, lambda b: b.body),
        ("body-face", lambda a: a.body, lambda b: b.face(index=4)),
        ("edge-body", lambda a: a.edge(index=2), lambda b: b.body),
    ]


def _occt_of(twin, shape):
    return twin.occt if shape is twin.body else twin.twin(shape)


@pytest.mark.parametrize("pair", pairs(), ids=lambda p: p[0])
def test_minimum_distances_match_occt(pair):
    a, b = twins()
    s1, s2 = pair[1](a), pair[2](b)
    ours = calc_distance(s1, s2, center=False)
    theirs = occt.calc_distance(_occt_of(a, s1), _occt_of(b, s2), center=False)
    assert ours["distance"] == pytest.approx(theirs["distance"], abs=1e-6)
    assert ours["info"] == "min"
    # the points are the distance apart and on the shapes
    assert np.linalg.norm(np.subtract(ours["point 2"], ours["point 1"])) == pytest.approx(ours["distance"], abs=1e-9)
    assert s1.distance_to(ours["point 1"]) < 1e-6 and s2.distance_to(ours["point 2"]) < 1e-6


@pytest.mark.parametrize("pair", pairs(), ids=lambda p: p[0])
def test_center_distances_match_occt(pair):
    a, b = twins()
    s1, s2 = pair[1](a), pair[2](b)
    ours = calc_distance(s1, s2, center=True)
    theirs = occt.calc_distance(_occt_of(a, s1), _occt_of(b, s2), center=True)
    assert ours["distance"] == pytest.approx(theirs["distance"], abs=1e-6)
    assert ours["info"] == "center"


def _segment_distance(a0, a1, b0, b1):
    """The distance of two straight segments, analytically (Ericson, Real-Time Collision
    Detection 5.1.9): an oracle for the alternating projection of two edges."""
    p1, q1, p2, q2 = (np.asarray(x, dtype=float) for x in (a0, a1, b0, b1))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    b, c = d1 @ d2, d1 @ r
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def test_straight_edges_match_the_analytic_distance_and_crossing_ones_are_zero_apart():
    _, _, edges, _ = load(serialize(Box(10, 10, 10)))
    _, _, other, _ = load(serialize(Box(10, 10, 10).moved(Rot(z=45))))
    crossing = 0
    for e in edges:
        for o in other:
            expected = _segment_distance(e.start, e.end, o.start, o.end)
            assert calc_distance(e, o)["distance"] == pytest.approx(expected, abs=1e-9)
            if expected < 1e-9:
                crossing += 1
    # an octagram at the top and one at the bottom: every side of a square crosses two of the rotated one
    assert crossing == 16


def test_angles_match_occt():
    a, b = twins()
    for s1, s2 in [
        (a.face(), b.face()),
        (a.edge(), b.edge()),
        (a.edge(), b.face()),
        (a.face("cylinder"), b.edge()),
    ]:
        ours, theirs = calc_angle(s1, s2), occt.calc_angle(_occt_of(a, s1), _occt_of(b, s2))
        assert ours["angle"] == pytest.approx(theirs["angle"], abs=1e-6)
        assert ours["reference 1"] == theirs["reference 1"] and ours["reference 2"] == theirs["reference 2"]
        assert set(ours) == set(theirs)


def test_a_vertex_has_no_angle():
    a, b = twins()
    assert calc_angle(a.vertices[0], b.face()) is None
    assert len(get_distance(a.vertices[0], b.face(), center=False)["result"]) == 2


def get_distance(shape1, shape2, center):
    """The distance tool's response for smlib shapes, as the backend assembles it."""
    return distance_response(shape1, shape2, center, calc_distance, calc_angle)


def test_the_distance_response_has_the_layout_of_the_occt_one():
    a, b = twins()
    ours = get_distance(a.face(), b.face(), center=False)
    theirs = occt.get_distance(_occt_of(a, a.face()), _occt_of(b, b.face()), center=False)
    assert [list(s.keys()) for s in ours["result"]] == [list(s.keys()) for s in theirs["result"]]
    assert ours["result"][2]["angle"] == theirs["result"][2]["angle"]


# ---------------------------------------------------------------------------
# through ocp_viewer_core's ViewerBackend
# ---------------------------------------------------------------------------


def test_the_model_goes_through_json_and_the_tools_answer():
    a = plate()
    b = Box(5, 5, 5).moved(Pos(40, 0, 0))
    mapping = {"parts": [entry("/scene/plate", a), {"parts": [entry("/scene/group/box", b)]}]}
    on_the_wire = wire(mapping)
    assert isinstance(on_the_wire["parts"][0]["shape"]["obj"], str)
    assert base64.b64decode(on_the_wire["parts"][0]["shape"]["obj"]) == serialize(a)

    viewer = ViewerBackend()
    viewer.load_model(on_the_wire)
    assert "/scene/plate" in viewer.model and "/scene/group/box/faces/faces_5" in viewer.model
    assert len([k for k in viewer.model if k.startswith("/scene/plate/edges/")]) == len(a.edges(seams=True))

    viewer.handle_event({"activeTool": "PropertiesMeasurement"}, MessageType.UPDATES)
    response = viewer.handle_event({"selectedShapeIDs": ["/scene/plate/faces/faces_0", "shift"]}, MessageType.UPDATES)
    assert response["tool_type"] == "PropertiesMeasurement" and response["shape_type"] == "Face"
    assert section(response, "area") == pytest.approx(a.faces()[0].area)

    viewer.handle_event({"activeTool": "DistanceMeasurement"}, MessageType.UPDATES)
    response = viewer.handle_event(
        {"selectedShapeIDs": ["/scene/plate/faces/faces_0", "/scene/group/box/faces/faces_0", False]},
        MessageType.UPDATES,
    )
    assert response["tool_type"] == "DistanceMeasurement"
    assert response["result"][0]["distance"] == pytest.approx(calc_distance(a.faces()[0], b.faces()[0])["distance"])
    assert orjson.dumps(response)  # JSON-clean


def test_a_vertex_entry_of_the_model_is_a_point():
    viewer = ViewerBackend()
    viewer.load_model(wire({"parts": [entry("/plate", plate())]}))
    vertex = viewer.model["/plate/vertices/vertices_0"]
    assert isinstance(vertex, Vertex) and isinstance(Vector(*tuple(vertex)), Vector)


# ---------------------------------------------------------------------------
# the whole way: show pipeline, wire, backend (what ocp_vscode does)
# ---------------------------------------------------------------------------


def shown(*objects, names):
    """What the show pipeline hands to the backend: the scene of smlib_show through
    ocp_tessellate, the mapping encoded as ocp_viewer_core's show sends it, through
    the codec and JSON, loaded by a backend. Returns the backend and the mesh's
    shapes, whose ids the viewer selects."""
    scene, names, colors, alphas = external_scene(objects, convert, names, None, None)
    group, instances = to_ocpgroup(*scene, names=names, colors=colors, alphas=alphas)
    _, shapes, mapping = tessellate_group(group, instances, {"deviation": DEVIATION, "angular_tolerance": ANGULAR})
    model = orjson.loads(orjson.dumps(encode_kernel_shapes(mapping), default=default))
    viewer = ViewerBackend()
    viewer.load_model(model)
    return viewer, shapes


def mesh_leaves(shapes):
    if "parts" in shapes:
        return [leaf for part in shapes["parts"] for leaf in mesh_leaves(part)]
    return [shapes]


def test_a_shown_part_is_measured_by_the_ids_of_its_mesh():
    part = plate()
    viewer, shapes = shown(part, Box(5, 5, 5).moved(Pos(40, 0, 0)), names=["shape0", "shape1"])
    leaves = {leaf["name"]: leaf for leaf in mesh_leaves(shapes)}
    plate_id, box_id = leaves["shape0"]["id"], leaves["shape1"]["id"]

    # every face and edge the viewer numbers has its shape in the backend
    for kind, count in (("faces", len(part.faces())), ("edges", len(part.edges(seams=True)))):
        for i in range(count):
            assert f"{plate_id}/{kind}/{kind}_{i}" in viewer.model
    assert viewer.plugins[plate_id].kernel == KERNEL

    viewer.handle_event({"activeTool": "DistanceMeasurement"}, MessageType.UPDATES)
    response = viewer.handle_event(
        {"selectedShapeIDs": [f"{plate_id}/edges/edges_9", f"{plate_id}/edges/edges_1", False]}, MessageType.UPDATES
    )
    _, _, edges, _ = load(serialize(part))
    assert response["result"][0]["distance"] == pytest.approx(calc_distance(edges[9], edges[1])["distance"])

    response = viewer.handle_event(
        {"selectedShapeIDs": [f"{plate_id}/faces/faces_3", f"{box_id}/faces/faces_0", False]}, MessageType.UPDATES
    )
    assert response["result"][0]["distance"] > 0 and orjson.dumps(response)


def test_a_part_in_a_moved_group_is_measured_where_it_is_drawn():
    moved = {"a": plate(), "b": Box(2, 2, 2)}
    viewer, shapes = shown(moved, names=["group"])
    for leaf in mesh_leaves(shapes):
        body = viewer.model[leaf["id"]]
        assert section(get_properties(body), "bb")["center"] == pytest.approx(tuple(moved[leaf["name"]].bbox().center))


def test_a_face_is_measured_as_the_shell_it_is_drawn_as():
    face = plate().faces()[6]
    viewer, shapes = shown(face, names=["face"])
    [leaf] = mesh_leaves(shapes)
    viewer.handle_event({"activeTool": "PropertiesMeasurement"}, MessageType.UPDATES)
    response = viewer.handle_event({"selectedShapeIDs": [f"{leaf['id']}/faces/faces_0", "shift"]}, MessageType.UPDATES)
    assert response["geom_type"] == "Cylinder" and section(response, "area") == pytest.approx(face.area)

