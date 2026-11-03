# ocp-viewer-smlibpy

Show [smlibpy](https://pypi.org/project/smlibpy/) objects (the SMLib kernel) in the OCP CAD Viewer family: [ocp_vscode](https://github.com/bernhard-42/vscode-ocp-cad-viewer), jupyter_cadquery, build123d_studio and ocp_viewer. The objects are tessellated by smlib's own kernel and go through [ocp-tessellate](https://github.com/bernhard-42/ocp-tessellate)'s plugin interface, so names, colors, the viewer's settings and `set_defaults` work as for build123d or CadQuery objects.

This package is the reference implementation of a kernel plugin for ocp-tessellate: `ocp_viewer_smlibpy/viewer.py` follows ocp-tessellate's `docs/ExternalKernel.md` step by step.

## Installation

```bash
pip install ocp-viewer-smlibpy
```

Requires Python 3.10+, smlibpy 0.6.2+, ocp-tessellate with the plugin interface (`ocp_tessellate.plugin`), ocp-viewer-core, and one OCCT binding (OCP or nanocct), which ocp-tessellate needs.

## Usage

```python
from smlibpy import *
from ocp_vscode import set_defaults, Camera
from ocp_viewer_smlibpy import smlib_show, smlib_show_all, to_occt

plate = Box(80, 60, 10)
plate = plate.fillet(plate.edges().parallel_to(Z), 8)
plate -= GridLocations(25, 16, 2, 3) * Cylinder(4, 10)

smlib_show(plate, plate.edges(), Pos(0, 0, 20), names=["plate", "edges", "loc"], reset_camera=Camera.RESET)
smlib_show_all()   # every smlibpy object in the current scope, under its variable name
```

`smlib_show` takes parts, sheets, sketches, faces, edges, wires, vertices, vectors, locations, planes and axes, and lists or dicts of them, with the `names`, `colors` and `alphas` lists and every keyword of the host's `show` (`reset_camera`, `collapse`, `helper_scale`, `transparent`, ...). Bodies are cached in ocp-tessellate's tessellation cache by their geometry.

### Measuring

The viewer's distance and properties tools ask a measurement backend (`ocp_viewer_core.backend.ViewerBackend`) that holds the geometry behind the mesh. `ocp_viewer_smlibpy/backend.py` is the smlib half of that, the backend contract of a kernel plugin: `serialize(body)` (the body as `.smb` bytes, what it travels as, since every host runs the backend in another process than the user's), `load(data, loc)` (the body rebuilt and placed, with its faces, edges and vertices in the order the viewer numbers them), `get_properties(shape)`, `calc_distance(shape1, shape2, center)` and `calc_angle(shape1, shape2)`, answering in the layout of `ocp_viewer_core.measure`. The package registers them under the kernel tag `smlib` when it is imported, through `ocp_viewer_core.backend.register_plugin`, and declares the entry point `smlib` of the group `ocp_viewer_core.backend`, through which the backend's process imports it in every host. The plugin must be installed where the backend runs: for VS Code in the interpreter the extension starts the backend with, for Jupyter CadQuery in the Jupyter server's environment. ocp-viewer-core's `ExternalKernel.md` describes the protocol. `convert` tags every body with the kernel, so ocp-tessellate puts it into the backend's model.

## Mixed scenes

`smlib_show` accepts smlibpy objects only. To show them beside build123d, CadQuery or OCCT objects, hand the converted objects to the host's `show` yourself:

```python
from ocp_viewer_smlibpy.viewer import convert
show(convert(plate, "plate"), build123d_box)
```

Each kernel's shapes are measured against each other, but a distance between an smlib shape and an OCCT shape is not: the kernels have no common geometry to measure on except OCCT itself. To measure across kernels, convert the smlib objects to OCCT shapes with `to_occt` and show them as OCCT objects:

```python
from ocp_viewer_smlibpy import to_occt
show(build123d_box, to_occt(plate), names=["box", "plate"])   # every distance between the two is measured
```

`to_occt` converts through smlib's `.brep` writer (bodies), its NURBS data (edges, wires) or its coordinates (vertices, locations, planes, axes), into the binding of `ocp_tessellate._kernel.kernel()` or the one named by `binding="OCP"` / `"nanocct"`. The converted object is OCCT's from then on: OCCT tessellates, numbers and measures it. The conversion fails for geometry smlib cannot write as `.brep`, offset surfaces of free-form faces, with a `ValueError` naming `smlib_show` as the way to at least see the object.

## Development

```bash
make lint        # ruff
make typecheck   # ty
make tests       # pytest
```

The tests need smlibpy and an OCCT binding installed; the comparison tests against the OCCT path build the same boxes with the binding.

## License

Apache 2.0
