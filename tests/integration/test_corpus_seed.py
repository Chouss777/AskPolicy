import re
from pathlib import Path

import pytest


POLICY_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "policies" / "fr"
TRAVEL_FILENAME = "TRAVEL-001-FR-V1.md"
LEAVE_FILENAME = "LEAVE-001-FR-V2.md"

# These are the required YAML spellings for this seed, including quoted dates,
# unquoted nulls, and flow lists. The checks only cover this front-matter format.
EXPECTED_FRONT_MATTER = {
    TRAVEL_FILENAME: {
        "policy_id": "TRAVEL-001",
        "document_id": "TRAVEL-001-FR-V1",
        "version": "1",
        "language": "fr",
        "title": "Politique des déplacements professionnels",
        "department": "Operations",
        "classification": "PUBLIC",
        "allowed_roles": "[employee, hr, finance, it_security]",
        "effective_date": '"2026-09-01"',
        "expires_at": "null",
        "supersedes": "null",
        "approval_status": "approved",
        "source_type": "synthetic_policy",
        "source_license": "project_owned",
    },
    LEAVE_FILENAME: {
        "policy_id": "LEAVE-001",
        "document_id": "LEAVE-001-FR-V2",
        "version": "2",
        "language": "fr",
        "title": "Politique des congés et absences",
        "department": "HR",
        "classification": "RESTRICTED",
        "allowed_roles": "[hr]",
        "effective_date": '"2026-09-01"',
        "expires_at": "null",
        "supersedes": "LEAVE-001-FR-V1",
        "approval_status": "approved",
        "source_type": "synthetic_policy",
        "source_license": "project_owned",
    },
}

PROHIBITED_MARKERS = ("TODO", "TBD", "XXX", "placeholder", "lorem ipsum", "[insert")


def _read_policy(filename):
    path = POLICY_DIRECTORY / filename
    assert path.is_file(), f"Missing seed policy: {path}"
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    assert lines[:1] == ["---"], f"Missing opening front-matter delimiter in {filename}"
    assert "---" in lines[1:], f"Missing closing front-matter delimiter in {filename}"
    closing_index = lines.index("---", 1)

    fields = {}
    for line in lines[1:closing_index]:
        key, separator, value = line.partition(": ")
        assert separator and re.fullmatch(r"[a-z_]+", key) and value.strip(), (
            f"Invalid front-matter line in {filename}: {line!r}"
        )
        assert key not in fields, f"Duplicate front-matter field {key!r} in {filename}"
        fields[key] = value

    body = "\n".join(lines[closing_index + 1:])
    return source, fields, body


@pytest.mark.parametrize("filename", EXPECTED_FRONT_MATTER)
def test_seed_policy_has_exact_front_matter(filename):
    _, fields, _ = _read_policy(filename)

    assert fields == EXPECTED_FRONT_MATTER[filename]


def test_seed_document_ids_are_distinct_and_match_filenames():
    document_ids = set()
    for filename in EXPECTED_FRONT_MATTER:
        _, fields, _ = _read_policy(filename)
        document_id = fields["document_id"]

        assert re.fullmatch(r"[A-Z]+-\d{3}-FR-V[1-9]\d*", document_id)
        assert filename == f"{document_id}.md"
        assert document_id not in document_ids
        document_ids.add(document_id)


@pytest.mark.parametrize("filename", EXPECTED_FRONT_MATTER)
def test_seed_policy_has_numbered_sections_and_a_markdown_table(filename):
    _, _, body = _read_policy(filename)
    sections = re.findall(r"^## ([1-9]\d*)\. \S", body, re.MULTILINE)
    assert len(sections) >= 5
    assert [int(number) for number in sections] == list(range(1, len(sections) + 1))

    table = re.search(
        r"^\|(?:[^|\n]+\|){2,}[ \t]*\n"
        r"\|(?:[ \t]*:?-{3,}:?[ \t]*\|){2,}[ \t]*\n"
        r"\|(?:[^|\n]+\|){2,}[ \t]*$",
        body,
        re.MULTILINE,
    )
    assert table is not None, f"Missing Markdown table in {filename}"
    assert len({line.count("|") for line in table.group().splitlines()}) == 1


def test_travel_policy_sets_a_numeric_accommodation_limit_in_mad():
    _, _, body = _read_policy(TRAVEL_FILENAME)
    limit = re.search(
        r"^\| Hébergement \| Plafond de ([1-9]\d*) MAD par nuit \|",
        body,
        re.MULTILINE,
    )
    assert limit is not None


@pytest.mark.parametrize(
    ("filename", "approval_rule", "exception_rule"),
    [
        (
            TRAVEL_FILENAME,
            "Si le coût prévisionnel total dépasse 3 000 MAD, le responsable "
            "hiérarchique et le service Finance doivent approuver la demande "
            "avant toute réservation.",
            "En cas d’urgence de sécurité, le voyageur peut réserver le transport "
            "nécessaire sans accord préalable.",
        ),
        (
            LEAVE_FILENAME,
            "Si une demande de congé ordinaire couvre plus de cinq jours ouvrés "
            "consécutifs, le responsable hiérarchique confirme la continuité de "
            "service et les RH valident le solde avant de notifier l’accord.",
            "En cas d’urgence familiale, les RH peuvent autoriser une absence sans "
            "validation préalable du responsable hiérarchique.",
        ),
    ],
    ids=["travel", "leave"],
)
def test_seed_policy_states_an_approval_condition_and_exception(
    filename, approval_rule, exception_rule
):
    _, _, body = _read_policy(filename)
    lines = body.splitlines()

    assert f"**Approbation conditionnelle.** {approval_rule}" in lines
    assert f"**Exception.** {exception_rule}" in lines


@pytest.mark.parametrize("filename", EXPECTED_FRONT_MATTER)
def test_seed_policy_contains_no_placeholder_markers(filename):
    source, _, _ = _read_policy(filename)
    normalized = source.casefold()
    found = [marker for marker in PROHIBITED_MARKERS if marker.casefold() in normalized]

    assert not found, f"Placeholder markers in {filename}: {found}"
