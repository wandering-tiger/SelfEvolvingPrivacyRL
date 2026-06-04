import os


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


TOOLS = {
    "read_file": ReadFileTool(),
    "list_dir": ListDirTool(),
}