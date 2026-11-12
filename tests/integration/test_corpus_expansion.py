import re
from pathlib import Path

import pytest


POLICY_ROOT = Path(__file__).resolve().parents[2] / "data" / "policies"
TRAVEL_EN = POLICY_ROOT / "en" / "TRAVEL-001-EN-V1.md"
TRAVEL_FR = POLICY_ROOT / "fr" / "TRAVEL-001-FR-V1.md"
EXPENSES_AR = POLICY_ROOT / "ar" / "EXPENSES-001-AR-V2.md"

# The required spellings include the quoted date, the explicit role list, and
# unquoted YAML null values. These checks are limited to the corpus structure.
COMMON_FIELDS = {
    "allowed_roles": "[employee, hr, finance, it_security]",
    "effective_date": '"2026-09-01"',
    "expires_at": "null",
    "supersedes": "null",
    "approval_status": "approved",
    "source_type": "synthetic_policy",
    "source_license": "project_owned",
}
EXPECTED_FIELDS = {
    TRAVEL_EN: {
        **COMMON_FIELDS,
        "policy_id": "TRAVEL-001",
        "document_id": "TRAVEL-001-EN-V1",
        "version": "1",
        "language": "en",
        "title": "Business Travel Policy",
        "department": "Operations",
        "classification": "PUBLIC",
    },
    EXPENSES_AR: {
        **COMMON_FIELDS,
        "policy_id": "EXPENSES-001",
        "document_id": "EXPENSES-001-AR-V2",
        "version": "2",
        "language": "ar",
        "title": "سياسة المصاريف المهنية",
        "department": "Finance",
        "classification": "INTERNAL",
    },
}
PROHIBITED_MARKERS = ("TODO", "TBD", "XXX", "placeholder", "lorem ipsum", "[insert")
ARABIC_WORD = re.compile(r"[\u0621-\u064A]{2,}")


def _read_policy(path):
    assert path.is_file(), f"Missing policy: {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:1] == ["---"], f"Missing opening front-matter delimiter in {path}"
    assert "---" in lines[1:], f"Missing closing front-matter delimiter in {path}"
    closing = lines.index("---", 1)

    fields = {}
    for line in lines[1:closing]:
        key, separator, value = line.partition(": ")
        assert separator and re.fullmatch(r"[a-z_]+", key) and value.strip(), (
            f"Invalid front-matter line in {path}: {line!r}"
        )
        assert key not in fields, f"Duplicate front-matter field {key!r} in {path}"
        fields[key] = value

    return fields, "\n".join(lines[closing + 1:])


@pytest.mark.parametrize("path", EXPECTED_FIELDS, ids=lambda path: path.stem)
def test_expanded_policy_has_exact_metadata_and_filename(path):
    fields, _ = _read_policy(path)

    assert len(fields) == 14
    assert fields == EXPECTED_FIELDS[path]
    assert path.name == f"{fields['document_id']}.md"


def test_expanded_document_ids_are_unique_across_the_corpus():
    document_paths = {}
    for path in sorted(POLICY_ROOT.rglob("*.md")):
        fields, _ = _read_policy(path)
        document_id = fields["document_id"].strip("'\"")
        assert document_id not in document_paths, (
            f"{path} and {document_paths[document_id]} share {document_id}"
        )
        document_paths[document_id] = path


@pytest.mark.parametrize("path", EXPECTED_FIELDS, ids=lambda path: path.stem)
def test_expanded_policy_has_numbered_sections_and_a_table(path):
    _, body = _read_policy(path)
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
    assert table is not None, f"Missing Markdown table in {path}"
    assert len({line.count("|") for line in table.group().splitlines()}) == 1


def test_travel_accommodation_limit_matches_the_french_policy():
    _, french_body = _read_policy(TRAVEL_FR)
    _, english_body = _read_policy(TRAVEL_EN)

    french_limit = re.search(
        r"^\| Hébergement \| Plafond de (\d+) MAD par nuit \|", french_body, re.MULTILINE
    )
    english_limit = re.search(
        r"^\| Accommodation \| Limit of (\d+) MAD per night \|", english_body, re.MULTILINE
    )
    assert french_limit is not None
    assert english_limit is not None
    assert french_limit.group(1) == english_limit.group(1) == "850"


def test_travel_dual_approval_threshold_matches_the_french_policy():
    _, french_body = _read_policy(TRAVEL_FR)
    _, english_body = _read_policy(TRAVEL_EN)

    french_rule = re.search(
        r"Si le coût prévisionnel total dépasse (\d[\d ]*) MAD, le responsable "
        r"hiérarchique et le service Finance doivent approuver la demande "
        r"avant toute réservation\.",
        french_body,
    )
    english_rule = re.search(
        r"If the total estimated cost exceeds (\d[\d,]*) MAD, both the line manager "
        r"and Finance must approve the request before any booking is made\.",
        english_body,
    )
    assert french_rule is not None
    assert english_rule is not None
    french_threshold = french_rule.group(1).replace(" ", "")
    english_threshold = english_rule.group(1).replace(",", "")
    assert french_threshold == english_threshold == "3000"


def test_english_travel_preserves_the_safety_emergency_exception():
    _, body = _read_policy(TRAVEL_EN)

    assert (
        "**Exception.** In a safety emergency, the traveler may book necessary "
        "transport without prior approval."
    ) in body.splitlines()


def test_arabic_expenses_states_approval_condition_and_exception():
    _, body = _read_policy(EXPENSES_AR)
    lines = body.splitlines()

    assert (
        "**الموافقة المشروطة.** إذا تجاوزت التكلفة التقديرية للمشتريات التشغيلية "
        "1,000 درهم مغربي (MAD)، وجب الحصول على موافقة المسؤول المباشر ومصلحة "
        "المالية قبل الالتزام بالنفقة."
    ) in lines
    assert (
        "**استثناء.** إذا تعذر الحصول على الإيصال الأصلي، يجوز لمصلحة المالية "
        "قبول نسخة صادرة عن المورد مع إثبات الدفع وبيان سبب تعذر تقديم الأصل."
    ) in lines


def test_arabic_expenses_has_arabic_policy_sections_and_a_mad_limit():
    _, body = _read_policy(EXPENSES_AR)
    sections = re.split(r"^## [1-9]\d*\. ", body, flags=re.MULTILINE)[1:]
    assert len(sections) >= 5
    for section in sections:
        heading, _, content = section.partition("\n")
        assert ARABIC_WORD.search(heading)
        assert ARABIC_WORD.search(content)

    assert re.search(
        r"^\| وجبة خلال مهمة ميدانية \| سقف [1-9]\d* درهمًا مغربيًا \(MAD\) لكل وجبة ",
        body,
        re.MULTILINE,
    )


@pytest.mark.parametrize("path", EXPECTED_FIELDS, ids=lambda path: path.stem)
def test_expanded_policy_contains_no_placeholder_markers(path):
    normalized = path.read_text(encoding="utf-8").casefold()
    found = [marker for marker in PROHIBITED_MARKERS if marker.casefold() in normalized]

    assert not found, f"Placeholder markers in {path}: {found}"
