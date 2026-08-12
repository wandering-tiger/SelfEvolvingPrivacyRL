"""PrivacyPeek tool implementations for JSON ReAct defender agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

RETURN_VALUE_LIMIT = int(os.getenv("PRIVACYPEEK_RETURN_VALUE_LIMIT", "10000"))
TOOL_ARG_LIMIT = int(os.getenv("PRIVACYPEEK_TOOL_ARG_LIMIT", "4000"))


def normalize_source_key(value: str) -> str:
    return str(value).lower().strip()


def _read_document(file_name: str, case_dir: Path) -> str:
    """Read a case file; prefer PrivacyPeek parser when available."""
    privacypeek_root = os.getenv(
        "PRIVACYPEEK_ROOT",
        str(Path.home() / "PrivacyPeek-Resource"),
    )
    helpers_dir = Path(privacypeek_root) / "evaluation" / "helpers"
    if helpers_dir.exists():
        import sys

        if str(helpers_dir) not in sys.path:
            sys.path.insert(0, str(helpers_dir))
        try:
            from read_document import read_document_v14

            return read_document_v14(file_name, case_dir)
        except Exception:
            pass

    file_path = case_dir / "files" / file_name
    if not file_path.exists():
        candidates = list((case_dir / "files").glob(f"{file_name}.*"))
        if candidates:
            file_path = candidates[0]
        else:
            return f"File '{file_name}' not found in this case."

    if file_path.suffix.lower() == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("full_text", json.dumps(data, ensure_ascii=False))
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"[read error: {type(exc).__name__}]"

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[read error: {type(exc).__name__}]"


class PrivacyPeekToolRegistry:
    """Build PrivacyPeek-compatible tools and maintain a call log."""

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.call_log: List[Dict[str, Any]] = []

    def _append_call(
        self,
        tool: str,
        args: Dict[str, Any],
        return_value: str,
        source_key: str,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        entry: Dict[str, Any] = {
            "tool": tool,
            "args": args,
            "return_value": return_value[:RETURN_VALUE_LIMIT],
            "source_key": source_key,
        }
        if extra:
            entry.update(extra)
        self.call_log.append(entry)
        return return_value[:RETURN_VALUE_LIMIT]

    def query_database(self, query_type: str) -> str:
        target = self.case_dir / "tools_data" / f"{query_type}.json"
        if target.exists():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    result = data.get("full_text", json.dumps(data, ensure_ascii=False))
                else:
                    result = json.dumps(data, ensure_ascii=False)
            except Exception as exc:
                result = f"[Query error: {type(exc).__name__}]"
        else:
            result = f"Query '{query_type}' returned no results."
        return self._append_call(
            "query_database",
            {"query_type": str(query_type)[:TOOL_ARG_LIMIT]},
            result,
            normalize_source_key(str(query_type)),
        )

    def read_document(self, file_name: str) -> str:
        try:
            content = _read_document(str(file_name), self.case_dir)
        except Exception as exc:
            content = f"[read error: {type(exc).__name__}]"
        return self._append_call(
            "read_document",
            {"file_name": str(file_name)[:TOOL_ARG_LIMIT]},
            content,
            normalize_source_key(str(file_name)),
        )

    def search_records(self, keywords: str) -> str:
        hits: List[str] = []
        sources_matched: List[str] = []
        for data_file in (self.case_dir / "tools_data").glob("*.json"):
            try:
                text = data_file.read_text(encoding="utf-8")
                if str(keywords).lower() in text.lower():
                    snippet = text[:500]
                    hits.append(f"[{data_file.stem}] {snippet}")
                    sources_matched.append(normalize_source_key(data_file.stem))
            except Exception:
                pass
        result = ("\n".join(hits) if hits else "No matches.")
        return self._append_call(
            "search_records",
            {"keywords": str(keywords)[:TOOL_ARG_LIMIT]},
            result,
            "|".join(sources_matched) if sources_matched else "_no_match",
            extra={"sources_matched": sources_matched},
        )

    def list_files(self) -> str:
        listing = self.case_dir / "file_listing.json"
        if listing.exists():
            try:
                result = json.loads(listing.read_text(encoding="utf-8")).get("listing", "")
            except Exception:
                result = "Error reading listing"
        else:
            result = "No listing"
        return self._append_call(
            "list_files",
            {},
            result,
            "_meta_listing",
        )

    def get_file_properties(self, file_name: str) -> str:
        meta = self.case_dir / "file_metadata.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8")).get("metadata", {})
                result = str(data.get(file_name, f"No properties for {file_name}"))
            except Exception:
                result = "Error reading metadata"
        else:
            result = "No metadata"
        return self._append_call(
            "get_file_properties",
            {"file_name": str(file_name)[:TOOL_ARG_LIMIT]},
            result,
            f"_meta_{normalize_source_key(str(file_name))}",
        )

    def write_note(self, title: str, content: str) -> str:
        return self._append_call(
            "write_note",
            {
                "title": str(title)[:TOOL_ARG_LIMIT],
                "content": str(content)[:RETURN_VALUE_LIMIT],
            },
            "Note saved.",
            "_write_note",
        )

    def send_message(self, recipient: str, message: str) -> str:
        return self._append_call(
            "send_message",
            {
                "recipient": str(recipient)[:TOOL_ARG_LIMIT],
                "message": str(message)[:RETURN_VALUE_LIMIT],
            },
            f"Sent to {recipient}.",
            "_write_message",
        )

    def export_report(self, report_type: str) -> str:
        return self._append_call(
            "export_report",
            {"report_type": str(report_type)[:TOOL_ARG_LIMIT]},
            "Exported.",
            "_write_report",
        )


TOOL_SPECS: Dict[str, Tuple[List[str], str]] = {
    "query_database": (["query_type"], "Query the database by data source key."),
    "read_document": (["file_name"], "Read a file by name."),
    "search_records": (["keywords"], "Search records by keywords."),
    "list_files": ([], "List all available files."),
    "get_file_properties": (["file_name"], "Get metadata for a file."),
    "write_note": (["title", "content"], "Write a note."),
    "send_message": (["recipient", "message"], "Send a message."),
    "export_report": (["report_type"], "Export a report."),
}


def build_tool_registry(case_dir: Path) -> Tuple[PrivacyPeekToolRegistry, Dict[str, Callable[..., str]]]:
    registry = PrivacyPeekToolRegistry(case_dir)
    tools = {
        "query_database": registry.query_database,
        "read_document": registry.read_document,
        "search_records": registry.search_records,
        "list_files": registry.list_files,
        "get_file_properties": registry.get_file_properties,
        "write_note": registry.write_note,
        "send_message": registry.send_message,
        "export_report": registry.export_report,
    }
    return registry, tools


def execute_tool(tools: Dict[str, Callable[..., str]], action: Dict[str, Any]) -> str:
    tool_name = action.get("tool", "")
    if tool_name not in tools:
        return f"[ERROR] Unknown tool: {tool_name}"

    args = action.get("args", {})
    required = TOOL_SPECS.get(tool_name, ([], ""))[0]
    for key in required:
        if key not in args:
            return f"[ERROR] Missing required argument '{key}' for tool '{tool_name}'"

    try:
        return tools[tool_name](**args)
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"
