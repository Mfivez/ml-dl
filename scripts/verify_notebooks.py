"""Validate course notebooks without changing them; optionally execute in memory.

Windows example (from the project root):
  .venv\\Scripts\\python.exe scripts/verify_notebooks.py . --days 1 2 3 4
  .venv\\Scripts\\python.exe scripts/verify_notebooks.py . --days 1 2 3 4 --execute

The caller must make this /tmp script available to Windows (copy to a temporary
Windows path or use the WSL UNC path). No notebook is written back to disk.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
import difflib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote, urlsplit

import nbformat


MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)")
HTML_IMAGE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EXERCISE_WORDS = ("à vous", "a vous", "exercice", "exercise", "extension", "solution")


def emit(message: str) -> None:
    print(message, flush=True)


def source_of(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def parse_python(source: str, filename: str) -> None:
    # IPython magics are legitimate notebook syntax. Transform only when needed.
    if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
        from IPython.core.inputtransformer2 import TransformerManager

        source = TransformerManager().transform_cell(source)
    ast.parse(source, filename=filename)


def check_notebook(path: Path, root: Path) -> tuple[object | None, list[str]]:
    problems: list[str] = []
    label = str(path.relative_to(root))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return None, [f"{label}: JSON: {error}"]

    ids: list[str] = []
    for number, cell in enumerate(raw.get("cells", []), start=1):
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not CELL_ID.fullmatch(cell_id):
            problems.append(f"{label}: cell {number}: missing/invalid cell id {cell_id!r}")
        else:
            ids.append(cell_id)
        source = source_of(cell)
        if cell.get("cell_type") == "code":
            try:
                parse_python(source, f"{label}:cell-{number}")
            except Exception as error:
                problems.append(f"{label}: cell {number}: Python: {error}")
        elif cell.get("cell_type") == "markdown":
            urls = MARKDOWN_IMAGE.findall(source) + HTML_IMAGE.findall(source)
            for url in urls:
                url = url.strip("<>")
                if url.startswith("attachment:"):
                    key = unquote(url.removeprefix("attachment:"))
                    if key not in cell.get("attachments", {}):
                        problems.append(f"{label}: cell {number}: missing attachment {key}")
                    continue
                parsed = urlsplit(url)
                if parsed.scheme or parsed.netloc:
                    continue  # Network assets are not downloaded by this verifier.
                target = path.parent / unquote(parsed.path)
                if not target.is_file():
                    problems.append(f"{label}: cell {number}: missing image {url}")

    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        problems.append(f"{label}: duplicate ids {duplicate_ids}")
    try:
        # Do not let nbformat auto-repair missing/duplicate ids before validation.
        notebook = nbformat.from_dict(raw)
        for cell in notebook.cells:
            cell.source = source_of(cell)
        nbformat.validate(notebook)
    except Exception as error:
        problems.append(f"{label}: nbformat: {error}")
        return None, problems
    return notebook, problems


def heading_lists(notebook: object) -> tuple[list[str], list[str]]:
    all_headings: list[str] = []
    theory_headings: list[str] = []
    exercise_active = False
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        # Only real Markdown headings, not text inside fenced code examples.
        fenced = False
        for line in cell.source.splitlines():
            if line.strip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            match = HEADING.match(line.strip())
            if not match:
                continue
            level, text = len(match.group(1)), match.group(2)
            lowered = text.casefold()
            if level == 1:
                # Titles can legitimately say "Corrigé" instead of "Exercice".
                exercise_active = "exercice" in lowered or "corrigé :" in lowered
                continue
            heading = "#" * level + " " + text
            all_headings.append(heading)
            if any(word in lowered for word in EXERCISE_WORDS):
                exercise_active = True
            if not exercise_active:
                theory_headings.append(heading)
    return all_headings, theory_headings


def check_pairs(root: Path, participants: list[Path], checked: dict) -> list[str]:
    problems: list[str] = []
    for support_path in participants:
        correction_path = (
            root / "corrections" / support_path.parent.name
            / (support_path.stem + "_corrige.ipynb")
        )
        if not correction_path.is_file():
            emit(f"[PAIR INFO] {support_path.relative_to(root)}: no correction notebook")
            continue
        support, correction = checked.get(support_path), checked.get(correction_path)
        if support is None or correction is None:
            continue
        all_support, theory_support = heading_lists(support)
        all_correction, theory_correction = heading_lists(correction)
        label = str(support_path.relative_to(root))
        if theory_support != theory_correction:
            diff = " | ".join(difflib.ndiff(theory_support, theory_correction))
            problems.append(f"{label}: theory headings differ: {diff}")
        else:
            heading_info = (
                "headings match" if all_support == all_correction
                else "theory headings match; exercise heading differences allowed"
            )
            emit(
                f"[PAIR OK] {label}: {heading_info}; "
                f"cells {len(support.cells)}/{len(correction.cells)}; "
                f"headings {len(all_support)}/{len(all_correction)}"
            )
        # Relative image paths differ because corrections live one level deeper.
        # Each path was independently checked above; textual path equality is not
        # a meaningful requirement. Exercise code is intentionally different.
    return problems


def execute_notebook(path: Path, notebook: object, root: Path, timeout: int) -> None:
    from nbclient import NotebookClient

    label = str(path.relative_to(root))
    start = time.monotonic()
    emit(f"[RUN] {label}")
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    output_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if output_errors:
        raise RuntimeError(f"{len(output_errors)} error outputs")
    for cell in code_cells:
        for output in cell.get("outputs", []):
            message = output.get("text", "") if output.output_type == "stream" else ""
            if any(word in message for word in ("accuracy validation", "Moyenne temporelle", "accuracy       ")):
                emit("[RESULT] " + message.strip())
    emit(f"[RUN OK] {label}: {len(code_cells)} code cells, {time.monotonic()-start:.1f}s")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Course repository root")
    parser.add_argument("--days", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--execute", action="store_true", help="Execute all selected notebooks in memory")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per cell in seconds (default 300)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if sys.platform == "win32":
        # Jupyter/ZMQ needs the selector policy on Windows rather than Proactor.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    paths: list[Path] = []
    participants: list[Path] = []
    issues: list[str] = []
    for day in dict.fromkeys(args.days):
        folder = root / f"jour_{day:02}"
        supports = sorted(folder.glob("*.ipynb"))
        if not supports:
            issues.append(f"No participant notebooks in {folder}")
        participants.extend(supports)
        paths.extend(supports)
        paths.extend(sorted((root / "corrections" / f"jour_{day:02}").glob("*.ipynb")))

    checked: dict = {}
    for path in paths:
        notebook, problems = check_notebook(path, root)
        checked[path] = notebook
        issues.extend(problems)
        if not problems:
            emit(f"[STRUCT OK] {path.relative_to(root)}")
    issues.extend(check_pairs(root, participants, checked))
    if issues:
        for problem in issues:
            emit(f"[ERROR] {problem}")
        emit(f"FAILED: {len(issues)} structural issues in {len(paths)} notebooks")
        return 1
    emit(f"STRUCTURE OK: {len(paths)} notebooks, days {','.join(map(str, args.days))}")

    if args.execute:
        failures: list[str] = []
        for path in paths:
            try:
                execute_notebook(path, checked[path], root, args.timeout)
            except Exception as error:
                label = str(path.relative_to(root))
                emit(f"[RUN ERROR] {label}: {error}")
                failures.append(label)
        if failures:
            emit(f"EXECUTION FAILED: {len(failures)}/{len(paths)}: {', '.join(failures)}")
            return 2
        emit(f"EXECUTION OK: {len(paths)} notebooks; no files modified")
    else:
        emit("Execution not requested; no notebooks were executed or modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
