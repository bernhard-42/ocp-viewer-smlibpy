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

"""What the two halves of the plugin agree on: `show` writes the bytes of a shape in the user's
process, `backend` reads them in the backend's, and importing either registers only that half."""

# the kernel tag of smlibpy's shapes, in the measurement backend's model and the registries
KERNEL = "smlib"

# the first byte is the format's version: the two processes can have other versions of this
# plugin. The second says how to read the rest: an .smp file holds bodies and curves, not points
FORMAT = 1
HEADER = bytes([FORMAT])
SMP, POINTS = b"S", b"V"
