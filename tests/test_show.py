"""The host's `show` and `show_all` take smlibpy objects directly, also at a debug
stop and beside objects of OCCT, and a script needs to import only smlibpy."""

import base64
import os
import subprocess
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import pytest
from ocp_tessellate.plugin import Placement, register_show
from ocp_viewer_core.comms import Comms, Session
from ocp_viewer_core.config import Config
from ocp_viewer_core.show import Viewer
from smlibpy import Axis, Box, Plane, Pos, Rot, Vector

import ocp_viewer_smlibpy
from ocp_viewer_smlibpy.show import SmlibShow
from ocp_viewer_smlibpy.tessellation import SmlibTessellation


class SinkComms(Comms[None]):
    def __init__(self):
        super().__init__()
        self.sent = []
        self.backend = []

    def workspace_config(self):
        return {"_splash": False}

    def status(self):
        return {}

    def send_data(self, data, port=None, timeit=False):
        self.sent.append(data)

    def send_backend(self, data, port=None, timeit=False):
        self.backend.append(data)


@pytest.fixture(name="viewer")
def _viewer():
    return Viewer(Config(Session(SinkComms()), ()))


def paths(viewer):
    def walk(node):
        yield node["id"]
        for part in node.get("parts", []):
            yield from walk(part)

    return list(walk(viewer.comms.sent[-1]["data"]["shapes"]))


def named(shown, name):
    return any(path.endswith(f"/{name}") for path in shown)


def test_the_two_show_functions_are_gone():
    assert not hasattr(ocp_viewer_smlibpy, "smlib_show")
    assert not hasattr(ocp_viewer_smlibpy, "smlib_show_all")


def test_show_takes_every_kind_of_smlibpy_object(viewer):
    box = Box(10, 10, 2)
    viewer.show(
        box,
        box.edges(),
        box.edges()[0],
        box.faces()[0],
        box.vertices()[0],
        Vector(1, 2, 3),
        Pos(0, 0, 5),
        Plane.XY,
        Axis.Z,
        names=["part", "edges", "edge", "face", "vertex", "vector", "location", "plane", "axis"],
    )
    shown = paths(viewer)
    for name in ("part", "edge", "face", "vertex", "vector", "location", "plane", "axis"):
        assert named(shown, name), (name, shown)
    assert sum("/edges/" in path for path in shown) == 12, shown


def test_a_part_goes_to_the_backend_tagged_with_its_kernel(viewer):
    box = Box(10, 10, 2)
    viewer.show(box, names=["part"])
    model = viewer.comms.backend[-1]["model"]
    entries = [part for part in model["parts"] if "shape" in part]
    assert entries[0]["shape"]["kernel"] == "smlib"


def test_show_takes_smlibpy_beside_build123d(viewer):
    bd = pytest.importorskip("build123d")
    viewer.show(Box(1, 1, 1), bd.Box(2, 2, 2), names=["smlib", "occt"])
    shown = paths(viewer)
    assert named(shown, "smlib") and named(shown, "occt")


def test_show_takes_a_list_of_both(viewer):
    bd = pytest.importorskip("build123d")
    viewer.show([Box(1, 1, 1), bd.Box(2, 2, 2)], names=["mixed"])
    shown = paths(viewer)
    assert named(shown, "Part") and named(shown, "Solid"), shown


def test_colors_and_alphas_reach_the_objects(viewer):
    viewer.show(Box(1, 1, 1), names=["part"], colors=["red"], alphas=[0.5])
    leaf = viewer.comms.sent[-1]["data"]["shapes"]["parts"][0]
    assert leaf["color"] == "#ff0000" and leaf["alpha"] == 0.5


def test_an_alpha_reaches_a_part_without_a_color(viewer):
    viewer.show(Box(1, 1, 1), names=["part"], alphas=[0.3])
    leaf = viewer.comms.sent[-1]["data"]["shapes"]["parts"][0]
    assert leaf["alpha"] == pytest.approx(0.3)


def axis_length(node):
    """The length of the first segment of the location's helper in the model."""
    if "parts" in node:
        for part in node["parts"]:
            found = axis_length(part)
            if found is not None:
                return found
        return None
    if "loc" in node.get("id", "") and node.get("type") == "edges":
        edges = node["shape"]["edges"]
        if isinstance(edges, dict):
            edges = np.frombuffer(base64.b64decode(edges["buffer"]), dtype=edges["dtype"])
        segment = np.asarray(edges).reshape(-1, 2, 3)[0]
        return float(np.linalg.norm(segment[1] - segment[0]))
    return None


BOX = Box(100, 100, 100)
# half the half diagonal of a box of 100, the helper_scale of 0.5 relative to it
RELATIVE = 0.5 * np.sqrt(3 * 50**2)


@pytest.mark.parametrize(
    "shapes",
    [BOX, [BOX], {"a": BOX}, BOX.edges(), list(BOX.edges()), BOX.edges()[0:12], BOX.faces()],
    ids=["part", "list", "dict", "ShapeList of edges", "list of edges", "slice of edges", "ShapeList of faces"],
)
def test_a_relative_helper_scale_measures_every_container(viewer, capsys, shapes):
    viewer.show(shapes, Pos(0, 0, 0), names=["shapes", "loc"], helper_scale=0.5)
    assert "Very small objects" not in capsys.readouterr().out
    assert abs(axis_length(viewer.comms.sent[-1]["data"]["shapes"]) - RELATIVE) < 1e-3


def test_a_plain_list_draws_the_same_as_a_shapelist(viewer):
    edges = BOX.edges()
    viewer.show(edges, names=["edges"])
    as_shapelist = sum("/edges/" in path for path in paths(viewer))
    viewer.show(list(edges), names=["edges"])
    assert sum("/edges/" in path for path in paths(viewer)) == as_shapelist == 12


def test_show_all_draws_the_scope_under_its_variable_names(viewer):
    box = Box(1, 1, 1)
    viewer.show_all({"box": box, "location": Pos(0, 0, 1), "edges": box.edges(), "number": 1, "text": "x"})
    shown = paths(viewer)
    assert named(shown, "box") and named(shown, "location") and named(shown, "edges")
    assert not named(shown, "number") and not named(shown, "text")


def test_show_all_at_a_debug_stop_draws_a_mixed_scope(viewer):
    bd = pytest.importorskip("build123d")
    viewer.show_all({"smlib": Box(1, 1, 1), "occt": bd.Box(2, 2, 2)}, _visual_debug=True)
    shown = paths(viewer)
    assert named(shown, "smlib") and named(shown, "occt")


def test_the_entry_points_are_declared_for_the_kernel_tag():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert '[project.entry-points."ocp_viewer.kernels.show"]\nsmlib = "ocp_viewer_smlibpy.show"\n' in pyproject
    assert '[project.entry-points."ocp_viewer.kernels.backend"]\nsmlib = "ocp_viewer_smlibpy.backend"\n' in pyproject


def _in_a_fresh_process(tmp_path, script):
    """With the entry points of pyproject.toml as an installed package would have them."""
    info = tmp_path / "entry-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: entry\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(
        "[ocp_viewer.kernels.show]\nsmlib = ocp_viewer_smlibpy.show\n\n"
        "[ocp_viewer.kernels.backend]\nsmlib = ocp_viewer_smlibpy.backend\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), *sys.path]))
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)


def test_a_script_that_imports_only_smlibpy_finds_the_plugin_for_drawing_only(tmp_path):
    result = _in_a_fresh_process(
        tmp_path,
        """
import sys
from smlibpy import Box
from ocp_tessellate.kernels import encode_kernel_shapes, is_external_object
assert "ocp_viewer_smlibpy" not in sys.modules
box = Box(1, 1, 1)
assert is_external_object(box) is True
model = encode_kernel_shapes({"parts": [{"id": "/b", "shape": {"kernel": "smlib", "obj": box}, "loc": None}]})
assert isinstance(model["parts"][0]["shape"]["obj"], str)
assert "ocp_viewer_smlibpy.show" in sys.modules
assert "ocp_viewer_smlibpy.backend" not in sys.modules
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Kernel plugin 'smlib' registered") == 1
    assert "registered for drawing" in result.stdout


def test_the_backend_finds_the_plugin_for_measuring_only(tmp_path):
    result = _in_a_fresh_process(
        tmp_path,
        """
import sys
from ocp_tessellate._show_registry import _show_plugins
from ocp_viewer_core.plugins import backend_of
backend_of("smlib")
assert "ocp_viewer_smlibpy.backend" in sys.modules
assert "ocp_viewer_smlibpy.show" not in sys.modules
assert "smlib" not in _show_plugins
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Kernel plugin 'smlib' registered") == 1
    assert "registered for measuring" in result.stdout


class Assembly:
    """What a kernel's assembly could be: its parts stay local, the parent holds their places."""

    def __init__(self, links, at=None):
        self.links, self.at = links, at


class AssemblyShow(SmlibShow):
    def is_object(self, obj):
        return isinstance(obj, Assembly)

    def get_children(self, obj):
        return obj.links

    def get_location(self, obj):
        return obj.at


register_show("smlib test assembly", SmlibTessellation(), AssemblyShow(), api=1)

Q0 = (0.0, 0.0, 0.0, 1.0)


def test_real_parts_in_an_assembly_land_where_it_says(viewer):
    sub = Assembly([("leg", Box(2, 2, 2), Placement((1.0, 0.0, 0.0), Q0))])
    top = Assembly([("bar", Box(2, 2, 2), Placement((0.0, 5.0, 0.0), Q0)), ("sub", sub, Placement((0.0, 10.0, 0.0), Q0))], at=Placement((100.0, 0.0, 0.0), Q0))
    viewer.show(top, names=["top"])
    shown = paths(viewer)
    assert "/top/bar" in shown and "/top/sub/leg" in shown
    box = viewer.comms.sent[-1]["data"]["shapes"]["bb"]
    assert (box["xmin"], box["xmax"]) == pytest.approx((99, 102))
    assert (box["ymin"], box["ymax"]) == pytest.approx((4, 11))


def test_a_smlibpy_location_places_a_child_through_from_axes(viewer):
    location = Pos(100, 0, 0) * Rot(0, 0, 90)
    placement = Placement.from_axes(tuple(location.position), tuple(location.x_dir), tuple(location.z_dir))
    viewer.show(Assembly([("bar", Box(2, 4, 2), None)], at=placement), names=["top"])
    box = viewer.comms.sent[-1]["data"]["shapes"]["bb"]
    assert (box["xmin"], box["xmax"], box["ymin"], box["ymax"]) == pytest.approx((98, 102, -1, 1))


def test_a_build123d_part_in_a_smlibpy_assembly(viewer):
    top = Assembly([("bar", Box(2, 2, 2), None), ("occt", bd.Box(2, 2, 2), Placement((0.0, 5.0, 0.0), Q0))], at=Placement((100.0, 0.0, 0.0), Q0))
    viewer.show(top, names=["top"])
    assert "/top/occt" in paths(viewer)
    box = viewer.comms.sent[-1]["data"]["shapes"]["bb"]
    assert (box["ymin"], box["ymax"], box["xmax"]) == pytest.approx((-1, 6, 101))
