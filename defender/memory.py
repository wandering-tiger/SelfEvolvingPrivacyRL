import os


class MemoryStore:

    def __init__(self):
        self.secret_file = "sandbox/secrets.txt"
        self.document_file = "sandbox/document.txt"
        self.workspace_dir = "sandbox/workspace"
        self._edited_files = set()

    # ---- Secrets ----
    def write_secret(self, secret):
        os.makedirs("sandbox", exist_ok=True)
        with open(self.secret_file, "w", encoding="utf-8") as f:
            f.write(secret)

    def get_secret_path(self):
        return self.secret_file

    # ---- Document (untrusted external content) ----
    def write_public(self, content):
        os.makedirs("sandbox", exist_ok=True)
        with open(self.document_file, "w", encoding="utf-8") as f:
            f.write(content)

    def get_document_path(self):
        return self.document_file

    # ---- Workspace (user task files) ----
    def write_workspace_file(self, filename, content):
        """Create a workspace file the user wants the agent to edit."""
        os.makedirs(self.workspace_dir, exist_ok=True)
        path = os.path.join(self.workspace_dir, os.path.basename(filename))
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def get_workspace_path(self, filename):
        return os.path.join(self.workspace_dir, os.path.basename(filename))

    def mark_workspace_edited(self, filename):
        self._edited_files.add(filename)

    def is_workspace_edited(self, filename):
        return filename in self._edited_files
