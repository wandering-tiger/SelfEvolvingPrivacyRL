import os


class WriteDocumentTool:
    """Write a malicious document to the attacker's sandbox.

    The document will be accessible to the defender agent as external content.
    """

    name = "write_document"

    def run(self, content: str) -> str:
        try:
            os.makedirs("attacker_sandbox", exist_ok=True)
            path = "attacker_sandbox/document.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Document written to {path}"
        except Exception as e:
            return f"[ERROR] {e}"


TOOLS = {
    "write_document": WriteDocumentTool(),
}
