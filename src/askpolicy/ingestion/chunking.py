"""Extract unpublished sections and chunks from AskPolicy Markdown."""

from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
import json
import re
from typing import Iterator, Literal

from .metadata import (
    Classification,
    Language,
    MetadataValidationError,
    PolicyMetadata,
    ReadRole,
    parse_policy_metadata,
)


BlockKind = Literal["heading", "paragraph", "procedure", "list", "table", "code", "thematic_break"]
_CHUNKING_FORMAT = "structural-blocks-v1"
_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*)|$)")
_FENCE = re.compile(r" {0,3}(\x60{3,}|~{3,})(.*)")
_LIST_ITEM = re.compile(r"( {0,3})([0-9]{1,9}[.)]|[-+*])(?:[ \t]+(.*)|$)")
_RULE = re.compile(r" {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})")
_SETEXT = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
_HTML_BLOCK = re.compile(r" {0,3}<(?:/?[a-zA-Z][\w-]*(?=[ \t/>]|$)|[!?])")
_BLOCK_QUOTE = re.compile(r" {0,3}>")
_SEPARATOR = re.compile(r":?-+:?")
# A pipe preceded by an even number of backslashes remains a delimiter.
_PIPE = re.compile(r"(?<!\\)(?:\\\\)*(\|)")


class ChunkExtractionError(ValueError):
    """Structural failure with a stable code and no source text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceRecord:
    document_id: str
    policy_id: str
    version: int
    language: Language
    source_path: str
    classification: Classification
    allowed_roles: tuple[ReadRole, ...]
    heading_path: tuple[str, ...]
    start: int
    end: int
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParentSection(_SourceRecord):
    """A heading and its direct content, up to the next heading at any level.

    A preamble has no heading or ancestor. Nested sections have their own
    spans and link to the nearest preceding heading of a lower level.
    """

    parent_id: str
    ancestor_id: str | None
    heading: str | None
    level: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildChunk(_SourceRecord):
    """One exact structural block owned by a single parent section."""

    child_id: str
    parent_id: str
    kind: BlockKind


def validate_parent_child(parent: ParentSection, child: ChildChunk) -> None:
    """Check a candidate link before using the parent as evidence."""
    if (
        child.parent_id != parent.parent_id
        or child.document_id != parent.document_id
        or child.policy_id != parent.policy_id
        or child.version != parent.version
        or child.language != parent.language
        or child.source_path != parent.source_path
        or child.classification != parent.classification
        or child.allowed_roles != parent.allowed_roles
        or child.heading_path != parent.heading_path
    ):
        raise ChunkExtractionError("INVALID_PARENT_CHILD")
    if not (0 <= parent.start <= child.start < child.end <= parent.end):
        raise ChunkExtractionError("INVALID_SOURCE_SPAN")
    if parent.end - parent.start != len(parent.text) or (
        parent.text[child.start - parent.start:child.end - parent.start] != child.text
    ):
        raise ChunkExtractionError("INVALID_SOURCE_SPAN")


@dataclass(frozen=True, slots=True)
class PolicyExtraction:
    """One canonical source and its ordered, unpublished records.

    Every span uses half-open Unicode character indices into canonical_source.
    """

    metadata: PolicyMetadata
    source_path: str
    source_sha256: str
    canonical_source: str = field(repr=False)
    body_start: int
    parents: tuple[ParentSection, ...]
    children: tuple[ChildChunk, ...]

    def __post_init__(self) -> None:
        parents: dict[str, ParentSection] = {}
        identities: set[str] = set()
        for record in (*self.parents, *self.children):
            identity = record.parent_id if isinstance(record, ParentSection) else record.child_id
            if identity in identities:
                raise ChunkExtractionError("DUPLICATE_CHUNK_ID")
            identities.add(identity)
            if not (
                self.body_start <= record.start < record.end <= len(self.canonical_source)
                and self.canonical_source[record.start:record.end] == record.text
            ):
                raise ChunkExtractionError("INVALID_SOURCE_SPAN")
            if isinstance(record, ParentSection):
                if (
                    record.document_id != self.metadata.document_id
                    or record.policy_id != self.metadata.policy_id
                    or record.version != self.metadata.version
                    or record.language != self.metadata.language
                    or record.source_path != self.source_path
                    or record.classification != self.metadata.classification
                    or record.allowed_roles != self.metadata.allowed_roles
                ):
                    raise ChunkExtractionError("INVALID_PARENT_CHILD")
                parents[record.parent_id] = record
            else:
                parent = parents.get(record.parent_id)
                if parent is None:
                    raise ChunkExtractionError("INVALID_PARENT_CHILD")
                validate_parent_child(parent, record)


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Block:
    kind: BlockKind
    start: int
    end: int


def _body_lines(source: str) -> tuple[int, list[_Line]]:
    lines = StringIO(source, newline="\n")
    offset = len(next(lines))
    # The production validator has already checked both delimiters.
    for raw in lines:
        offset += len(raw)
        if raw.removesuffix("\n") == "---":
            break
    body_start = offset
    body = []
    for raw in lines:
        body.append(_Line(offset, offset + len(raw), raw.removesuffix("\n")))
        offset += len(raw)
    return body_start, body


def _heading(text: str) -> tuple[int, str]:
    match = _HEADING.fullmatch(text)
    assert match is not None
    title = (match[2] or "").rstrip(" \t")
    if re.fullmatch(r"#+", title):
        title = ""
    else:
        title = re.sub(r"[ \t]+#+$", "", title)
    return len(match[1]), title


def _fenced_end(lines: list[_Line], start: int, opening: re.Match[str]) -> int:
    marker, info = opening.groups()
    if marker[0] == "\x60" and "\x60" in info:
        raise ChunkExtractionError("MALFORMED_STRUCTURE")
    close = re.compile(rf" {{0,3}}{marker[0]}{{{len(marker)},}}[ \t]*")
    for index in range(start + 1, len(lines)):
        if close.fullmatch(lines[index].text):
            return index + 1
    raise ChunkExtractionError("MALFORMED_STRUCTURE")


def _table_cells(text: str) -> list[str]:
    text = text.strip(" \t")
    pipes = [match.start(1) for match in _PIPE.finditer(text)]
    if not pipes:
        return []
    bounds = [-1, *pipes, len(text)]
    cells = [text[left + 1:right].strip(" \t") for left, right in zip(bounds, bounds[1:])]
    if pipes[0] == 0:
        cells = cells[1:]
    if pipes[-1] == len(text) - 1:
        cells = cells[:-1]
    return cells


def _table_starts(text: str, following: str) -> bool:
    cells = _table_cells(following)
    return text.lstrip(" ").startswith("|") or bool(
        cells and any(_SEPARATOR.fullmatch(cell) for cell in cells)
    )


def _explicit_block(text: str) -> bool:
    return bool(
        _HEADING.fullmatch(text) or _FENCE.fullmatch(text) or _RULE.fullmatch(text)
        or _LIST_ITEM.fullmatch(text) or _HTML_BLOCK.match(text) or _BLOCK_QUOTE.match(text)
    )


def _table_end(lines: list[_Line], start: int) -> int:
    header = _table_cells(lines[start].text)
    index = start + 1
    separator = _table_cells(lines[index].text) if index < len(lines) else []
    if (
        not header
        or all(_SEPARATOR.fullmatch(cell) for cell in header)
        or len(separator) != len(header)
        or not all(_SEPARATOR.fullmatch(cell) for cell in separator)
    ):
        raise ChunkExtractionError("INVALID_TABLE")
    index += 1
    while index < len(lines):
        text = lines[index].text
        if not text.strip() or _explicit_block(text):
            break
        cells = _table_cells(text)
        if len(cells) != len(header) or all(_SEPARATOR.fullmatch(cell) for cell in cells):
            raise ChunkExtractionError("INVALID_TABLE")
        index += 1
    return index


def _list_end(lines: list[_Line], start: int, opening: re.Match[str]) -> int:
    content = opening[3] or ""
    following = lines[start + 1].text if start + 1 < len(lines) else ""
    if _explicit_block(content) or _table_starts(content, following):
        raise ChunkExtractionError("UNSUPPORTED_STRUCTURE")

    end = index = start + 1
    indentation = len(opening[1])
    marker = opening[2][-1]
    content_indent = opening.start(3) if opening[3] is not None else opening.end(2) + 1
    while index < len(lines):
        text = lines[index].text
        if not text.strip():
            index += 1
            continue

        item = _LIST_ITEM.fullmatch(text)
        if item and len(item[1]) >= indentation and not _RULE.fullmatch(text):
            if len(item[1]) == indentation:
                if item[2][-1] != marker:
                    break
                content_indent = item.start(3) if item[3] is not None else item.end(2) + 1
            content = item[3] or ""
        elif text.startswith(" " * content_indent):
            content = text[content_indent:]
        elif index == end and not _explicit_block(text) and not text.lstrip(" ").startswith("|"):
            content = text
        else:
            break

        following = lines[index + 1].text if index + 1 < len(lines) else ""
        if (
            _FENCE.fullmatch(content)
            or _HEADING.fullmatch(content)
            or _HTML_BLOCK.match(content)
            or _BLOCK_QUOTE.match(content)
            or _table_starts(content, following)
        ):
            raise ChunkExtractionError("UNSUPPORTED_STRUCTURE")
        end = index + 1
        index = end
    return end


def _blocks(lines: list[_Line]) -> Iterator[_Block]:
    index = 0
    while index < len(lines):
        text = lines[index].text
        if not text.strip():
            index += 1
            continue

        start = index
        following = lines[index + 1].text if index + 1 < len(lines) else ""
        fence = _FENCE.fullmatch(text)
        item = _LIST_ITEM.fullmatch(text)
        kind: BlockKind
        if _HEADING.fullmatch(text):
            kind, index = "heading", index + 1
        elif fence is not None:
            kind, index = "code", _fenced_end(lines, index, fence)
        elif _RULE.fullmatch(text):
            kind, index = "thematic_break", index + 1
        elif _HTML_BLOCK.match(text) or _BLOCK_QUOTE.match(text):
            raise ChunkExtractionError("UNSUPPORTED_STRUCTURE")
        elif item is not None:
            kind = "procedure" if item[2][0].isdigit() else "list"
            index = _list_end(lines, index, item)
        elif text.startswith(("    ", "\t")):
            kind, index = "code", index + 1
            probe = index
            while probe < len(lines):
                next_line = lines[probe].text
                if next_line.strip():
                    if not next_line.startswith(("    ", "\t")):
                        break
                    index = probe + 1
                probe += 1
        elif _table_starts(text, following):
            kind, index = "table", _table_end(lines, index)
        else:
            kind, index = "paragraph", index + 1
            while index < len(lines):
                current = lines[index].text
                following = lines[index + 1].text if index + 1 < len(lines) else ""
                if _SETEXT.fullmatch(current):
                    raise ChunkExtractionError("UNSUPPORTED_STRUCTURE")
                if not current.strip() or _explicit_block(current) or _table_starts(current, following):
                    break
                index += 1
        yield _Block(kind, lines[start].start, lines[index - 1].end)


def _record_id(namespace: str, *parts: object) -> str:
    canonical = json.dumps([_CHUNKING_FORMAT, namespace, *parts], ensure_ascii=False, separators=(",", ":"))
    return f"{namespace}:{sha256(canonical.encode('utf-8')).hexdigest()}"


def extract_policy_chunks(markdown: str, *, source_path: str) -> PolicyExtraction:
    """Validate and extract one Markdown source with deterministic structural spans.

    ATX headings, paragraphs, simple lists, fenced/indented code and pipe
    tables are handled locally. Tables require consistent column counts.
    Code is opaque text. Raw HTML blocks, block quotes, Setext headings and
    complex structures nested in lists are outside the supported subset.

    LF and CRLF inputs produce the same records. Offsets count characters in
    the complete LF-normalized source, including the front matter. IDs hash a
    versioned structural format, document identity, authorization metadata,
    heading context, canonical span and exact text in separate namespaces.
    """
    source = markdown.replace("\r\n", "\n")
    metadata = parse_policy_metadata(source, source_path=source_path)
    if source_path is None:
        raise MetadataValidationError("INVALID_SOURCE_PATH")
    if "\r" in source:
        raise ChunkExtractionError("INVALID_SOURCE_NEWLINES")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError:
        raise ChunkExtractionError("UNSUPPORTED_STRUCTURE") from None

    body_start, lines = _body_lines(source)
    if "\0" in source[body_start:]:
        raise ChunkExtractionError("UNSUPPORTED_STRUCTURE")
    blocks = list(_blocks(lines))
    if not blocks:
        raise ChunkExtractionError("MALFORMED_STRUCTURE")

    boundaries = [0, *(i for i, block in enumerate(blocks) if i and block.kind == "heading"), len(blocks)]
    origin = {
        "document_id": metadata.document_id,
        "policy_id": metadata.policy_id,
        "version": metadata.version,
        "language": metadata.language,
        "source_path": source_path,
        "classification": metadata.classification,
        "allowed_roles": metadata.allowed_roles,
    }
    parents = []
    children = []
    ancestors: list[ParentSection] = []
    for first, stop in zip(boundaries, boundaries[1:]):
        first_block = blocks[first]
        if first_block.kind == "heading":
            level, heading = _heading(source[first_block.start:first_block.end].removesuffix("\n"))
            while ancestors and ancestors[-1].level >= level:
                ancestors.pop()
            ancestor_id = ancestors[-1].parent_id if ancestors else None
            heading_path = (*(parent.heading for parent in ancestors), heading)
        else:
            level, heading, ancestor_id, heading_path = 0, None, None, ()

        start = body_start if first == 0 and heading is None else first_block.start
        end = blocks[stop].start if stop < len(blocks) else len(source)
        text = source[start:end]
        parent_id = _record_id("parent", origin, heading_path, level, start, end, text)
        parent = ParentSection(
            **origin, heading_path=heading_path, parent_id=parent_id, ancestor_id=ancestor_id,
            heading=heading, level=level, start=start, end=end, text=text,
        )
        parents.append(parent)
        if level:
            ancestors.append(parent)
        for block in blocks[first:stop]:
            text = source[block.start:block.end]
            children.append(ChildChunk(
                **origin, heading_path=heading_path, parent_id=parent_id,
                child_id=_record_id("child", parent_id, block.kind, block.start, block.end, text),
                kind=block.kind, start=block.start, end=block.end, text=text,
            ))

    return PolicyExtraction(
        metadata=metadata, source_path=source_path, source_sha256=sha256(source_bytes).hexdigest(),
        canonical_source=source, body_start=body_start, parents=tuple(parents), children=tuple(children),
    )
