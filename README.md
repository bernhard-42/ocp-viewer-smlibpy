# ocp-viewer-smlibpy

Show [smlibpy](https://pypi.org/project/smlibpy/) objects (the SMLib kernel) in the OCP CAD Viewer family: [ocp_vscode](https://github.com/bernhard-42/vscode-ocp-cad-viewer), jupyter_cadquery, build123d_studio and ocp_viewer. The objects are tessellated by smlib's own kernel and go through [ocp-tessellate](https://github.com/bernhard-42/ocp-tessellate)'s plugin interface, so names, colors, the viewer's settings and `set_defaults` work as for build123d or CadQuery objects.

This package is the reference implementation of a kernel plugin for ocp-tessellate: one class per protocol of [CAD kernel plugins](https://bernhard-42.github.io/ocp_viewer_docs/plugins/overview/), `SmlibTessellation` in `tessellation.py`, `SmlibShow` in `show.py` and `SmlibBackend` in `backend.py`.

## Installation

```bash
pip install ocp-viewer-smlibpy
```

Requires Python 3.10+, smlibpy 0.6.2+, ocp-tessellate with the plugin interface (`ocp_tessellate.plugin`), ocp-viewer-core, and one OCCT binding (OCP or nanocct), which ocp-tessellate needs.

## Usage

```python
from smlibpy import *
from ocp_vscode import show, show_all, Camera

plate = Box(80, 60, 10)
plate = plate.fillet(plate.edges().parallel_to(Z), 8)
plate -= GridLocations(25, 16, 2, 3) * Cylinder(4, 10)

show(plate, plate.edges(), Pos(0, 0, 20), names=["plate", "edges", "loc"], reset_camera=Camera.RESET)
show_all()   # every smlibpy object in the current scope, under its variable name
```

The script imports only smlibpy and the host: ocp_tessellate imports the plugin's drawing half, `ocp_viewer_smlibpy.show`, through the entry point `smlib` of the group `ocp_viewer.kernels.show` at the first show. The host's `show` and `show_all` take parts, sheets, sketches, faces, edges, wires, vertices, vectors, locations, planes and axes, and lists or dicts of them, also at a debug stop (visual debugging), with the `names`, `colors` and `alphas` lists and every keyword of `show` (`reset_camera`, `collapse`, `helper_scale`, `transparent`, ...). Bodies are cached in ocp-tessellate's tessellation cache by their geometry.

### Measuring

The viewer's distance and properties tools ask a measurement backend (`ocp_viewer_core.backend.ViewerBackend`) that holds the geometry behind the mesh. `SmlibBackend` in `ocp_viewer_smlibpy/backend.py` is the smlib half of that, its implementation of `BackendProtocol`: `deserialize(data)` (the bytes of `SmlibShow.serialize`: bodies and curves as binary `.smp`, vertices as their coordinates, what they travel as, since every host runs the backend in another process than the user's), `moved(shape, placement)`, `parts(shape)` (its faces, edges and vertices in the order the viewer numbers them), `kind(shape)`, `center(shape)`, `properties(shape)` (what smlib knows about it, as plain numbers), `closest_points(shape1, shape2)` and `closest_directions(shape1, shape2)`. ocp_viewer_core runs that sequence and makes the panel's rows, as it does for OCCT. Importing `ocp_viewer_smlibpy.backend` registers it under the kernel tag `smlib`, through `ocp_viewer_core.backend.register_plugin`, and the entry point `smlib` of the group `ocp_viewer.kernels.backend` imports that module in the backend's process, in every host. The plugin must be installed where the backend runs: for VS Code in the interpreter the extension starts the backend with, for Jupyter CadQuery in the Jupyter server's environment. [The backend protocol](https://bernhard-42.github.io/ocp_viewer_docs/plugins/backend/) describes it. Every leaf of a shown smlibpy object has an entry in the backend's model, tagged `smlib` by ocp-tessellate.

## Mixed scenes

`show` and `show_all` take smlibpy objects beside build123d, CadQuery or OCCT objects:

```python
show(plate, build123d_box, names=["plate", "box"])
```

Each kernel's shapes are measured against each other, but a distance between an smlib shape and an OCCT shape is not: the kernels have no common geometry to measure on except OCCT itself. To measure across kernels, convert the smlib objects to OCCT shapes with `to_occt` and show them as OCCT objects:

```python
from ocp_viewer_smlibpy import to_occt
show(build123d_box, to_occt(plate), names=["box", "plate"])   # every distance between the two is measured
```

`to_occt` converts through smlib's `.brep` writer (bodies), its NURBS data (edges, wires) or its coordinates (vertices, locations, planes, axes), into the binding of `ocp_tessellate._kernel.kernel()` or the one named by `binding="OCP"` / `"nanocct"`. The converted object is OCCT's from then on: OCCT tessellates, numbers and measures it. The conversion fails for geometry smlib cannot write as `.brep`, offset surfaces of free-form faces, with a `ValueError` naming the host's `show` as the way to at least see the object.

## Development

```bash
make lint        # ruff
make typecheck   # ty
make tests       # pytest
```

The tests need smlibpy and an OCCT binding installed; the comparison tests against the OCCT path build the same boxes with the binding.

## License

Apache 2.0
