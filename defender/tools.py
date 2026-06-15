import os


SANDBOX_ROOT = os.path.abspath("sandbox")


def _is_safe_path(path: str) -> bool:
    """Ensure path stays within the sandbox directory."""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(SANDBOX_ROOT + os.sep) or abs_path == SANDBOX_ROOT


class ReadFileTool:

    name = "read_file"

    def run(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[ERROR] {e}"


class ListDirTool:

    name = "list_dir"

    def run(self, path):
        try:
            return "\n".join(os.listdir(path))
        except Exception as e:
            return f"[ERROR] {e}"


class WriteFileTool:
    """Write content to a file within the sandbox."""

    name = "write_file"

    def run(self, path, content):
        try:
            if not _is_safe_path(path):
                return f"[ERROR] Access denied: path outside sandbox ({path})"
            os.makedirs(os.path.dirname(path) or SANDBOX_ROOT, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {path} ({len(content)} chars)"
        except Exception as e:
            return f"[ERROR] {e}"


TOOLS = {
    "read_file": ReadFileTool(),
    "list_dir": ListDirTool(),
    "write_file": WriteFileTool(),
}
