"""The tests build the viewer's messages without a viewer: ocp_viewer_core.viewer
picks ocp_vscode as the host when VS Code's environment is present, and
ocp_vscode's pytest switch pins its port."""

import os

os.environ.setdefault("VSCODE_CWD", os.getcwd())
os.environ.setdefault("OCP_VSCODE_PYTEST", "1")
