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
READ_ROLES = {"employee", "hr", "finance", "it_security"}


def _parse_roles(value):
    block_style = isinstance(value, list)
    if block_style:
        items = value
    else:
        flow_list = re.fullmatch(r"\[([^\[\]]*)\](?:[ \t]+#.*)?", value.strip())
        assert flow_list is not None, f"Malformed allowed_roles list: {value!r}"
        contents = flow_list[1].strip().removesuffix(",")
        items = contents.split(",") if contents.strip() else []

    roles = []
    for item in items:
        if block_style:
            item = re.split(r"[ \t]+#", item, maxsplit=1)[0]
        token = re.fullmatch(r"""(['"]?)([a-z_]+)\1""", item.strip())
        assert token is not None, f"Malformed role grant: {item!r}"
        assert token[2] in READ_ROLES, f"Invalid read role: {token[2]!r}"
        roles.append(token[2])
    assert roles, "Empty allowed_roles list"
    assert len(roles) == len(set(roles)), "Duplicate read roles"
    return f"[{', '.join(roles)}]" if block_style else value


def _parse_policy(source, path):
    # Phase 2 front matter uses single-line scalars and a flow or block role list.
    lines = source.splitlines()
    assert lines[:1] == ["---"], f"Missing opening front-matter delimiter in {path}"
    assert "---" in lines[1:], f"Missing closing front-matter delimiter in {path}"
    closing = lines.index("---", 1)

    fields = {}
    key = None
    role_indent = None
    for line in lines[1:closing]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.fullmatch(r"( *)-[ \t]+(.+)", line)
        if item:
            assert key == "allowed_roles" and isinstance(fields[key], list), (
                f"Unexpected role-list item in {path}: {line!r}"
            )
            if role_indent is None:
                role_indent = item[1]
            assert item[1] == role_indent, f"Inconsistent role-list indentation in {path}"
            fields[key].append(item[2])
            continue

        field = re.fullmatch(r"([a-z_]+):(?:[ \t]+(.*))?", line)
        assert field is not None, f"Invalid front-matter line in {path}: {line!r}"
        key = field[1]
        assert key not in fields, f"Duplicate front-matter field {key!r} in {path}"
        value = field[2] or ""
        if key == "allowed_roles" and (not value.strip() or value.startswith("#")):
            fields[key] = []
        else:
            assert value.strip() and not value.startswith("#"), f"Empty {key!r} in {path}"
            fields[key] = value

    missing = EXPECTED_FIELDS[TRAVEL_EN].keys() - fields.keys()
    assert not missing, f"Missing required metadata in {path}: {sorted(missing)}"
    fields["allowed_roles"] = _parse_roles(fields["allowed_roles"])

    identifier = re.fullmatch(
        r"""(?:"([^"]+)"|'([^']+)'|([^\s"'\[\]{},#]+))""",
        fields["document_id"].strip(),
    )
    assert identifier is not None, f"Invalid document_id in {path}"
    document_id = identifier[1] or identifier[2] or identifier[3]
    if identifier[3] is not None:
        assert (
            document_id.casefold() not in {"null", "~", "true", "false"}
            and not document_id.isdecimal()
        ), f"Invalid document_id in {path}"
    assert path.name == f"{document_id}.md", f"Filename does not match document_id in {path}"

    return fields, "\n".join(lines[closing + 1:])


def _read_policy(path):
    assert path.is_file(), f"Missing policy: {path}"
    return _parse_policy(path.read_text(encoding="utf-8"), path)


def _policy_source(role_field="allowed_roles: [employee, hr]"):
    fields = [
        role_field if key == "allowed_roles" else f"{key}: {value}"
        for key, value in EXPECTED_FIELDS[TRAVEL_EN].items()
    ]
    return "\n".join(["---", *fields, "---", "# Fixture"])


@pytest.mark.parametrize(
    "role_field",
    [
        "allowed_roles: [employee, hr]",
        "allowed_roles:\n  - employee\n  - hr",
        "allowed_roles:\n- employee\n- hr",
    ],
    ids=["flow", "indented-block", "indentless-block"],
)
def test_policy_reader_accepts_supported_role_lists(role_field):
    fields, body = _parse_policy(_policy_source(role_field), TRAVEL_EN)

    assert fields == {**EXPECTED_FIELDS[TRAVEL_EN], "allowed_roles": "[employee, hr]"}
    assert body == "# Fixture"


@pytest.mark.parametrize(
    "role_field",
    ["allowed_roles: []", "allowed_roles:"],
    ids=["empty-flow", "empty-block"],
)
def test_policy_reader_rejects_empty_role_lists(role_field):
    with pytest.raises(AssertionError, match="Empty allowed_roles list"):
        _parse_policy(_policy_source(role_field), TRAVEL_EN)


@pytest.mark.parametrize(
    ("role_field", "error"),
    [
        ("allowed_roles: [ingestion_admin]", "Invalid read role"),
        ("allowed_roles:\n  - contractor", "Invalid read role"),
        ("allowed_roles: [employee, employee]", "Duplicate read roles"),
    ],
    ids=["admin-role", "unknown-role", "duplicate-role"],
)
def test_policy_reader_rejects_invalid_role_grants(role_field, error):
    with pytest.raises(AssertionError, match=error):
        _parse_policy(_policy_source(role_field), TRAVEL_EN)


@pytest.mark.parametrize(
    ("role_field", "error"),
    [
        ("allowed_roles: employee", "Malformed allowed_roles list"),
        ("allowed_roles: [employee, hr", "Malformed allowed_roles list"),
        ("allowed_roles: [employee]\n  - hr", "Unexpected role-list item"),
        ("allowed_roles:\n  - [employee, hr]", "Malformed role grant"),
        ("allowed_roles:\n  - employee\n    - hr", "Inconsistent role-list indentation"),
        ("allowed_roles:\n  - employee: hr", "Malformed role grant"),
    ],
    ids=["scalar", "unclosed-flow", "mixed-styles", "nested-list", "indentation", "mapping"],
)
def test_policy_reader_rejects_malformed_role_lists(role_field, error):
    with pytest.raises(AssertionError, match=error):
        _parse_policy(_policy_source(role_field), TRAVEL_EN)


def test_policy_reader_rejects_duplicate_metadata_keys():
    source = _policy_source("allowed_roles:\n  - employee\nallowed_roles: [hr]")

    with pytest.raises(AssertionError, match="Duplicate front-matter field 'allowed_roles'"):
        _parse_policy(source, TRAVEL_EN)


def test_policy_reader_rejects_missing_required_metadata():
    source = _policy_source().replace("title: Business Travel Policy\n", "", 1)

    with pytest.raises(AssertionError, match="Missing required metadata.*title"):
        _parse_policy(source, TRAVEL_EN)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("[TRAVEL-001-EN-V1]", "Invalid document_id"),
        ("OTHER-001-EN-V1", "Filename does not match document_id"),
    ],
    ids=["non-scalar", "filename-mismatch"],
)
def test_policy_reader_rejects_invalid_document_ids(value, error):
    source = _policy_source().replace("document_id: TRAVEL-001-EN-V1", f"document_id: {value}", 1)

    with pytest.raises(AssertionError, match=error):
        _parse_policy(source, TRAVEL_EN)


@pytest.mark.parametrize(
    ("delimiter", "replacement", "position"),
    [
        ("---\n", "----\n", "opening"),
        ("\n---\n", "\n----\n", "closing"),
    ],
    ids=["opening", "closing"],
)
def test_policy_reader_rejects_invalid_front_matter_delimiters(delimiter, replacement, position):
    source = _policy_source().replace(delimiter, replacement, 1)

    with pytest.raises(AssertionError, match=f"Missing {position} front-matter delimiter"):
        _parse_policy(source, TRAVEL_EN)


@pytest.mark.parametrize("path", EXPECTED_FIELDS, ids=lambda path: path.stem)
def test_expanded_policy_has_exact_metadata_and_filename(path):
    fields, _ = _read_policy(path)

    assert len(fields) == 14
    assert fields == EXPECTED_FIELDS[path]
    assert path.name == f"{fields['document_id']}.md"


def test_expanded_document_ids_are_unique_across_the_corpus():
    document_paths = {}
    for path in sorted(POLICY_ROOT.rglob("*.md")):
        _read_policy(path)
        document_id = path.stem
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
