from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def chunking():
    # package=false in uv. Import through the normal package entry point.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion import chunking as module

        yield module


MANIFEST = json.loads((ROOT / "data/corpus_manifest.yaml").read_text(encoding="utf-8"))["documents"]
FIXTURE_PATH = "data/policies/fr/FIXTURE-999-FR-V2.md"
FIXTURE_HEADER = """---
policy_id: FIXTURE-999
document_id: FIXTURE-999-FR-V2
version: 2
language: fr
title: Révision de la politique
department: Operations
classification: RESTRICTED
allowed_roles: [hr]
effective_date: "2026-09-01"
expires_at: null
supersedes: FIXTURE-999-FR-V1
approval_status: approved
source_type: synthetic_policy
source_license: project_owned
---
"""
FIXTURE_BODY = """Avant le titre, café et بيانات.

# Politique ###
Introduction.

## 1. Demande
Condition d’approbation : accord préalable.

### Détails
1. Première étape.

   Suite avec indentation.
2. Deuxième étape.

- Élément A
  - Sous-élément
- Élément B

| Élément | Valeur |
| :--- | ---: |
| Accès \\| données | بيانات |
| Révision | Oui |

**Exception.** Approbation spéciale.

###### Profondeur
Texte profond.

## 2. Suite
Texte final.
"""


@pytest.fixture
def valid_source(chunking):
    source = FIXTURE_HEADER + FIXTURE_BODY
    result = chunking.extract_policy_chunks(source, source_path=FIXTURE_PATH)
    assert result.children
    return source, result


def _assert_spans_and_coverage(module, result):
    canonical = result.canonical_source
    parent_by_id = {parent.parent_id: parent for parent in result.parents}
    ids = [*parent_by_id, *(child.child_id for child in result.children)]
    assert len(set(ids)) == len(result.parents) + len(result.children)
    for records in (result.parents, result.children):
        end = result.body_start
        for record in records:
            assert end <= record.start < record.end <= len(canonical)
            assert not canonical[end:record.start].strip()
            assert canonical[record.start:record.end] == record.text
            end = record.end
        assert not canonical[end:].strip()
    for child in result.children:
        module.validate_parent_child(parent_by_id[child.parent_id], child)


@pytest.mark.parametrize("record", MANIFEST, ids=[row["document_id"] for row in MANIFEST])
def test_corpus_extraction_preserves_source_spans_metadata_and_tables(chunking, record):
    path = record["source_path"]
    source = (ROOT / path).read_bytes().decode("utf-8")
    canonical = source.replace("\r\n", "\n")
    result = chunking.extract_policy_chunks(source, source_path=path)

    assert result.canonical_source == canonical
    assert result.source_sha256 == record["content_sha256"]
    assert len(result.metadata.model_dump()) == 14
    assert result.metadata.model_dump(mode="json") == {
        key: value for key, value in record.items() if key not in {"source_path", "content_sha256"}
    }
    assert result.body_start == canonical.index("\n---\n", 4) + len("\n---\n")
    assert all(parent.parent_id.startswith("parent:") for parent in result.parents)
    assert all(child.child_id.startswith("child:") for child in result.children)
    for item in (*result.parents, *result.children):
        for field in ("document_id", "policy_id", "version", "language", "classification"):
            assert getattr(item, field) == record[field]
        assert item.source_path == path
        assert item.allowed_roles == tuple(record["allowed_roles"])
    _assert_spans_and_coverage(chunking, result)

    body = canonical[result.body_start:]
    expected_tables = [
        block for block in body.split("\n\n") if block.startswith("|")
    ]
    assert expected_tables, record["document_id"]
    assert [child.text.removesuffix("\n") for child in result.children if child.kind == "table"] == [
        table.removesuffix("\n") for table in expected_tables
    ]


def test_independent_extractions_are_deterministic_in_either_corpus_order(chunking):
    assert len(MANIFEST) == 26

    def extract(records):
        return {
            row["document_id"]: chunking.extract_policy_chunks(
                (ROOT / row["source_path"]).read_bytes().decode("utf-8"),
                source_path=row["source_path"],
            )
            for row in records
        }

    forward = extract(MANIFEST)
    backward = extract(reversed(MANIFEST))
    assert forward == backward
    parent_ids = [parent.parent_id for result in forward.values() for parent in result.parents]
    child_ids = [child.child_id for result in forward.values() for child in result.children]
    assert len(set(parent_ids + child_ids)) == len(parent_ids) + len(child_ids)
    for prefix in ("LEAVE-001", "ACCESS-001", "TRAVEL-001"):
        samples = [result for key, result in forward.items() if key.startswith(prefix)]
        assert samples
        expected = {
            "LEAVE-001": ("hr",),
            "ACCESS-001": ("it_security",),
            "TRAVEL-001": ("employee", "hr", "finance", "it_security"),
        }[prefix]
        assert all(child.allowed_roles == expected for result in samples for child in result.children)


def test_hierarchy_preamble_and_unicode_character_offsets(chunking, valid_source):
    source, result = valid_source
    parents = result.parents
    assert [parent.level for parent in parents] == [0, 1, 2, 3, 6, 2]
    assert [parent.heading_path for parent in parents] == [
        (), ("Politique",), ("Politique", "1. Demande"),
        ("Politique", "1. Demande", "Détails"),
        ("Politique", "1. Demande", "Détails", "Profondeur"),
        ("Politique", "2. Suite"),
    ]
    assert [parent.ancestor_id for parent in parents] == [
        None, None, parents[1].parent_id, parents[2].parent_id, parents[3].parent_id, parents[1].parent_id,
    ]
    assert parents[0].text == "Avant le titre, café et بيانات.\n\n"
    assert parents[1].text.startswith("# Politique ###\n")
    assert parents[3].start == source.index("### Détails")
    assert len(source[:parents[3].start].encode("utf-8")) > parents[3].start
    assert chunking.extract_policy_chunks(
        source.replace("\n", "\r\n"), source_path=FIXTURE_PATH
    ) == result
    _assert_spans_and_coverage(chunking, result)
    with pytest.raises(FrozenInstanceError):
        parents[0].text = "Modified"


def test_procedure_list_table_approval_and_exception_are_intact(valid_source):
    _, result = valid_source
    procedure = next(child for child in result.children if child.kind == "procedure")
    assert procedure.text == "1. Première étape.\n\n   Suite avec indentation.\n2. Deuxième étape.\n"
    bullets = next(child for child in result.children if child.kind == "list")
    assert bullets.text == "- Élément A\n  - Sous-élément\n- Élément B\n"
    table = next(child for child in result.children if child.kind == "table")
    assert table.text == "| Élément | Valeur |\n| :--- | ---: |\n| Accès \\| données | بيانات |\n| Révision | Oui |\n"
    approval = next(child for child in result.children if child.text.startswith("Condition d’approbation"))
    exception = next(child for child in result.children if child.text.startswith("**Exception.**"))
    assert approval.heading_path == ("Politique", "1. Demande")
    assert exception.parent_id == procedure.parent_id == table.parent_id
    assert exception.heading_path[-1] == "Détails"


def test_fenced_headings_are_opaque_source_text(chunking, valid_source):
    source, _ = valid_source
    fence = "\x60" * 4
    code = f"{fence}md\n## Faux titre\n| opaque |\n{fence[:3]}\n{fence}\n"
    result = chunking.extract_policy_chunks(source + code + "\n## 3. Réel\n", source_path=FIXTURE_PATH)
    assert all(parent.heading != "Faux titre" for parent in result.parents)
    assert result.parents[-1].heading == "3. Réel"
    assert [child.text for child in result.children if child.kind == "code"] == [code]
    _assert_spans_and_coverage(chunking, result)


def test_changed_content_changes_affected_identities(chunking, valid_source):
    source, result = valid_source
    revised = chunking.extract_policy_chunks(
        source.replace("Texte final.", "Texte autre.", 1), source_path=FIXTURE_PATH
    )
    assert result.source_sha256 != revised.source_sha256
    assert result.parents[:-1] == revised.parents[:-1]
    assert result.parents[-1].parent_id != revised.parents[-1].parent_id
    assert result.children[-1].child_id != revised.children[-1].child_id
    assert result.children[-1].start == revised.children[-1].start
    assert result.children[-1].end == revised.children[-1].end


def test_table_without_outer_pipes_preserves_columns(chunking, valid_source):
    source, _ = valid_source
    table = "Colonne A | Colonne B\n--- | :---:\nalpha | beta\n"
    result = chunking.extract_policy_chunks(source + "\n" + table, source_path=FIXTURE_PATH)
    assert result.children[-1].kind == "table"
    assert result.children[-1].text == table


@pytest.mark.parametrize(
    ("before", "after", "code"),
    [
        ("allowed_roles: [hr]\n", "", "MISSING_ALLOWED_ROLES"),
        ("allowed_roles: [hr]", "allowed_roles: [ingestion_admin]", "INVALID_ALLOWED_ROLES"),
        ('effective_date: "2026-09-01"\n', "", "MISSING_EFFECTIVE_DATE"),
        ("title: Révision de la politique", "title: !tag secret", "MALFORMED_FRONT_MATTER"),
    ],
)
def test_metadata_failures_are_preserved(chunking, valid_source, before, after, code):
    source, _ = valid_source
    assert source.count(before) == 1
    with pytest.raises(chunking.MetadataValidationError) as error:
        chunking.extract_policy_chunks(source.replace(before, after, 1), source_path=FIXTURE_PATH)
    assert error.value.code == code


def test_unsafe_source_path_uses_the_metadata_error(chunking, valid_source):
    source, _ = valid_source
    with pytest.raises(chunking.MetadataValidationError) as error:
        chunking.extract_policy_chunks(source, source_path=f"../{FIXTURE_PATH}")
    assert error.value.code == "INVALID_SOURCE_PATH"


@pytest.mark.parametrize(
    ("before", "after", "code"),
    [
        ("Texte final.", "\x60\x60\x60\nTexte final.", "MALFORMED_STRUCTURE"),
        ("Texte final.", "<script>private content</script>", "UNSUPPORTED_STRUCTURE"),
        ("Texte final.", "Texte\r final.", "INVALID_SOURCE_NEWLINES"),
        ("| :--- | ---: |", "| broken | ---: |", "INVALID_TABLE"),
        ("| Révision | Oui |", "| Révision |", "INVALID_TABLE"),
        ("| Révision | Oui |", "Révision Oui", "INVALID_TABLE"),
        ("Texte final.", "Autre titre\n===========", "UNSUPPORTED_STRUCTURE"),
        ("Texte final.", "> ## Titre cité", "UNSUPPORTED_STRUCTURE"),
        ("- Élément A", "- ## Titre imbriqué", "UNSUPPORTED_STRUCTURE"),
        ("| Révision | Oui |", "| --- | --- |", "INVALID_TABLE"),
    ],
    ids=[
        "fence", "html", "standalone-cr", "separator", "short-row", "row-without-pipes",
        "setext", "blockquote", "nested-heading", "duplicate-separator",
    ],
)
def test_structural_errors_are_named_and_hide_input(chunking, valid_source, before, after, code):
    source, _ = valid_source
    assert source.count(before) == 1
    with pytest.raises(chunking.ChunkExtractionError) as error:
        chunking.extract_policy_chunks(source.replace(before, after, 1), source_path=FIXTURE_PATH)
    assert error.value.code == code
    assert str(error.value) == code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_id", "parent:forged"),
        ("document_id", "OTHER-999-FR-V2"),
        ("policy_id", "OTHER-999"),
        ("version", 1),
        ("language", "en"),
        ("source_path", "data/policies/fr/OTHER-999-FR-V2.md"),
        ("classification", "PUBLIC"),
        ("allowed_roles", ("employee", "hr")),
        ("heading_path", ("Other section",)),
    ],
)
def test_parent_child_rejects_inconsistent_fields(chunking, valid_source, field, value):
    _, result = valid_source
    parent, child = result.parents[0], result.children[0]
    chunking.validate_parent_child(parent, child)
    with pytest.raises(chunking.ChunkExtractionError) as error:
        chunking.validate_parent_child(parent, replace(child, **{field: value}))
    assert error.value.code == "INVALID_PARENT_CHILD"


@pytest.mark.parametrize("field", ["start", "end", "text"])
def test_parent_child_rejects_invalid_span_or_slice(chunking, valid_source, field):
    _, result = valid_source
    parent, child = result.parents[0], result.children[0]
    chunking.validate_parent_child(parent, child)
    invalid = {"start": parent.start - 1, "end": parent.end + 1, "text": "Wrong text"}
    with pytest.raises(chunking.ChunkExtractionError) as error:
        chunking.validate_parent_child(parent, replace(child, **{field: invalid[field]}))
    assert error.value.code == "INVALID_SOURCE_SPAN"


@pytest.mark.parametrize("kind", ["parents", "children"])
def test_extraction_rejects_duplicate_ids(chunking, valid_source, kind):
    _, result = valid_source
    records = getattr(result, kind)
    with pytest.raises(chunking.ChunkExtractionError) as error:
        replace(result, **{kind: (*records, records[0])})
    assert error.value.code == "DUPLICATE_CHUNK_ID"


def test_extraction_rejects_parent_with_acl_different_from_validated_metadata(chunking, valid_source):
    _, result = valid_source
    parent = replace(result.parents[0], allowed_roles=("hr", "employee"))
    with pytest.raises(chunking.ChunkExtractionError) as error:
        replace(result, parents=(parent, *result.parents[1:]))
    assert error.value.code == "INVALID_PARENT_CHILD"


@pytest.mark.parametrize(
    "foreign_id",
    ["LEAVE-001-FR-V1", "LEAVE-001-EN-V2", "ACCESS-001-FR-V2"],
    ids=["cross-version", "cross-language", "cross-document"],
)
def test_real_policy_parents_cannot_be_reused_across_boundaries(chunking, foreign_id):
    def extract(document_id):
        path = next(row["source_path"] for row in MANIFEST if row["document_id"] == document_id)
        return chunking.extract_policy_chunks((ROOT / path).read_text(encoding="utf-8"), source_path=path)

    current = extract("LEAVE-001-FR-V2")
    other = extract(foreign_id)
    parent, child = current.parents[0], current.children[0]
    assert other.parents[0].parent_id != parent.parent_id
    assert other.children[0].child_id != child.child_id
    chunking.validate_parent_child(parent, child)
    with pytest.raises(chunking.ChunkExtractionError) as error:
        chunking.validate_parent_child(other.parents[0], replace(child, parent_id=other.parents[0].parent_id))
    assert error.value.code == "INVALID_PARENT_CHILD"
