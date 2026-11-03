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

"""smlibpy objects in the OCP CAD Viewer family: the kernel plugin for SMLib.

`smlib_show` and `smlib_show_all` show parts, sheets, sketches, faces, edges,
wires, vertices, locations, planes and axes of smlibpy, tessellated by smlib's
own kernel, through ocp_tessellate's plugin interface (docs/ExternalKernel.md of
ocp-tessellate). The viewer's measurement tools answer for them from smlib's own
geometry: importing this package registers `backend` with ocp_viewer_core's
measurement backend, and the entry point `smlib` of the group
`ocp_viewer_core.backend` (pyproject.toml) imports it in the backend's process,
which every host runs apart from the user's. `to_occt` converts them to OCCT shapes.
"""

from ocp_viewer_core.backend import register_plugin

from . import backend
from ._version import __version__ as __version__, __version_info__ as __version_info__
from .viewer import smlib_show, smlib_show_all, to_occt

__all__ = ["__version__", "__version_info__", "backend", "smlib_show", "smlib_show_all", "to_occt"]

register_plugin(
    backend.KERNEL,
    backend.load,
    backend.serialize,
    backend.get_properties,
    backend.calc_distance,
    backend.calc_angle,
)
