## v0.1.0 (unreleased)

**Features**

- `smlib_show` and `smlib_show_all`: smlibpy parts, sheets, sketches, faces, edges, wires, vertices, vectors, locations, planes and axes in the OCP CAD Viewer family, tessellated by smlib's kernel through ocp-tessellate's plugin interface (`ocp_tessellate.plugin`), with names, colors, alphas, every `show` keyword and `set_defaults`, and the tessellation cache keyed by the body's geometry
- Edges sampled by chord deviation: the segment count is estimated from the curvature and refined where the estimate falls short; edge polylines stay within a tenth of the face deviation
- `to_occt`: smlibpy objects as OCCT shapes of OCP or nanocct, to measure them against OCCT objects: `show(occt_obj, to_occt(smlib_obj))`, since the backend measures the shapes of one kernel against each other only
- `ocp_viewer_smlibpy.backend`, the backend contract of the plugin: `serialize`, `load`, `get_properties`, `calc_distance` and `calc_angle` answer the viewer's measurement tools from smlib's own geometry (distances between any two of bodies, faces, edges and vertices, the angle of edges and faces at their closest points, radii and centers of circles, cylinders, cones, spheres and tori), in the layout of `ocp_viewer_core.measure`. Importing the package registers it with `ocp_viewer_core.backend.register_plugin` under the kernel tag `smlib`, the entry point `smlib` of the group `ocp_viewer_core.backend` imports it in the backend's process, and `convert` tags every body, so ocp-tessellate puts it into the backend's model
