import os


class MemoryStore:

    def __init__(self):

        self.secret_file = "sandbox/secrets.txt"
        self.document_file = "sandbox/document.txt"

    def write_secret(self, secret):

        os.makedirs("sandbox", exist_ok=True)

        with open(
            self.secret_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(secret)

    def write_public(
        self,
        content,
    ):

        with open(
            self.document_file,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

    def get_secret_path(self):

        return self.secret_file

    def get_document_path(self):

        return self.document_file