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

The host's `show` and `show_all` take parts, sheets, sketches, faces, edges, wires,
vertices, locations, planes and axes of smlibpy, tessellated by smlib's own kernel and
measured from its own geometry. The plugin is three classes, one per protocol:

- `tessellation.SmlibTessellation`: ocp_tessellate's TessellationProtocol
- `show.SmlibShow`: ocp_tessellate's ShowProtocol
- `backend.SmlibBackend`: ocp_viewer_core's BackendProtocol

Importing `show` registers the first two, importing `backend` the third. The entry points
`smlib` of the groups `ocp_viewer.kernels.show` and `ocp_viewer.kernels.backend`
(pyproject.toml) import them, the first in the user's process at the first show, so a script
needs only `from smlibpy import ...`, the second in the backend's process, which every host
runs apart from the user's. `to_occt` converts smlibpy objects to OCCT shapes.
"""

from ._version import __version__ as __version__, __version_info__ as __version_info__
from .occt import to_occt

__all__ = ["__version__", "__version_info__", "to_occt"]
