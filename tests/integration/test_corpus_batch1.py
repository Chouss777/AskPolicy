import json
import re
from datetime import date
from pathlib import Path

import pytest

from .test_corpus_expansion import PROHIBITED_MARKERS, _read_policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = REPOSITORY_ROOT / "data" / "policies"
MATRIX_PATH = REPOSITORY_ROOT / "docs" / "corpus-rule-matrix.md"

# title, department, classification, explicit read roles, supersedes
BATCH_IDENTITIES = {
    "REMOTE-001-FR-V2": ("Politique de télétravail", "HR", "INTERNAL", "[employee, hr, finance, it_security]", "REMOTE-001-FR-V1"),
    "EXPENSES-001-FR-V2": ("Politique des dépenses professionnelles", "Finance", "INTERNAL", "[employee, hr, finance, it_security]", "EXPENSES-001-FR-V1"),
    "ACCESS-001-FR-V2": ("Politique de contrôle d'accès", "IT Security", "RESTRICTED", "[it_security]", "ACCESS-001-FR-V1"),
    "DATA-001-FR-V2": ("Politique de gestion des données", "IT Security", "INTERNAL", "[employee, hr, finance, it_security]", "DATA-001-FR-V1"),
    "INCIDENT-001-FR-V1": ("Politique de signalement des incidents", "IT Security", "INTERNAL", "[employee, hr, finance, it_security]", "null"),
    "PROCUREMENT-001-FR-V1": ("Politique des achats", "Finance", "INTERNAL", "[employee, hr, finance, it_security]", "null"),
    "LEAVE-001-EN-V2": ("Leave and Absence Policy", "HR", "RESTRICTED", "[hr]", "null"),
    "EXPENSES-001-EN-V2": ("Professional Expenses Policy", "Finance", "INTERNAL", "[employee, hr, finance, it_security]", "null"),
}
SEED_IDS = {
    "TRAVEL-001-FR-V1",
    "TRAVEL-001-EN-V1",
    "LEAVE-001-FR-V2",
    "EXPENSES-001-AR-V2",
}
REQUIRED_IDS = SEED_IDS | BATCH_IDENTITIES.keys()
REQUIRED_FIELDS = {
    "policy_id", "document_id", "version", "language", "title", "department",
    "classification", "allowed_roles", "effective_date", "expires_at",
    "supersedes", "approval_status", "source_type", "source_license",
}
CLASSIFICATIONS = {"PUBLIC", "INTERNAL", "RESTRICTED"}
READ_ROLES = {"employee", "hr", "finance", "it_security"}

# Full-scope slots from docs/decisions.md §3, using the established family IDs.
ACTIVE_SLOTS = {
    (family, language, version)
    for family, version, languages in (
        ("REMOTE-001", 2, ("fr", "en", "ar")),
        ("LEAVE-001", 2, ("fr", "en", "ar")),
        ("EXPENSES-001", 2, ("fr", "en", "ar")),
        ("TRAVEL-001", 1, ("fr", "en")),
        ("ACCESS-001", 2, ("fr", "en")),
        ("DATA-001", 2, ("fr", "en", "ar")),
        ("INCIDENT-001", 1, ("fr", "en")),
        ("PROCUREMENT-001", 1, ("fr", "en")),
    )
    for language in languages
}
HISTORICAL_IDS = {
    ("REMOTE-001", "fr", 1): "REMOTE-001-FR-V1",
    ("REMOTE-001", "en", 1): "REMOTE-001-EN-V1",
    ("EXPENSES-001", "fr", 1): "EXPENSES-001-FR-V1",
    ("ACCESS-001", "fr", 1): "ACCESS-001-FR-V1",
    ("DATA-001", "fr", 1): "DATA-001-FR-V1",
    ("LEAVE-001", "fr", 1): "LEAVE-001-FR-V1",
}
AUTHORIZED_SLOTS = ACTIVE_SLOTS | HISTORICAL_IDS.keys()
ALL_READ_ROLES = "[employee, hr, finance, it_security]"
AMOUNT = r"([0-9][0-9, \u00a0\u202f]*)"


def _policy_path(document_id):
    _, language, _ = document_id.rsplit("-", 2)
    return POLICY_ROOT / language.lower() / f"{document_id}.md"


def _read_future_policy(path):
    # Phase 2 structural reader. It supports the corpus' top-level scalars and
    # flow or block role lists. General YAML validation belongs to Phase 3.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:1] == ["---"] and "---" in lines[1:], f"Invalid front matter in {path}"
    closing = lines.index("---", 1)
    fields = {}
    key = None
    for line in lines[1:closing]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.fullmatch(r"[ \t]*-[ \t]+(.+)", line)
        if item:
            assert key == "allowed_roles" and isinstance(fields[key], list), path
            fields[key].append(item[1])
            continue
        field = re.fullmatch(r"([a-z_]+):(?:[ \t]+(.*))?", line)
        assert field is not None, f"Unsupported front-matter line in {path}: {line!r}"
        key = field[1]
        assert key not in fields, f"Duplicate front-matter field {key!r} in {path}"
        value = (field[2] or "").strip()
        if key == "allowed_roles" and (not value or value.startswith("#")):
            fields[key] = []
        else:
            fields[key] = value
    return fields, "\n".join(lines[closing + 1:])


def _scalar(value):
    assert isinstance(value, str), f"Expected a scalar, got {value!r}"
    value = value.strip()
    if value.startswith('"'):
        value = json.loads(value)
    elif value.startswith("'"):
        assert re.fullmatch(r"'(?:[^']|'')*'", value), value
        value = value[1:-1].replace("''", "'")
    else:
        value = re.split(r"[ \t]+#", value, maxsplit=1)[0].strip()
        assert not value.startswith(("[", "{", "|", ">", "&", "*", "!")), value
    assert value.strip(), "Empty metadata scalar"
    return value


def _nullable(value):
    decoded = _scalar(value)
    if not value.lstrip().startswith(("'", '"')) and decoded in {"null", "Null", "NULL", "~"}:
        return None
    return decoded


def _read_roles(value):
    if isinstance(value, str):
        roles = re.fullmatch(r"\[([^\n]*)\](?:[ \t]+#.*)?", value)
        assert roles is not None, f"Invalid allowed_roles: {value!r}"
        value = roles[1].split(",") if roles[1].strip() else []
    assert isinstance(value, list), f"Expected a role list, got {value!r}"
    roles = [_scalar(role) for role in value]
    assert roles and len(roles) == len(set(roles)), "Missing or duplicate read roles"
    assert set(roles) <= READ_ROLES, f"Unknown read role in {roles!r}"
    return set(roles)


@pytest.fixture(scope="module")
def corpus():
    required_paths = {_policy_path(document_id) for document_id in REQUIRED_IDS}
    paths = sorted(path for path in POLICY_ROOT.rglob("*") if path.is_file())
    assert all(path.suffix == ".md" for path in paths), "Unexpected file in the policy tree"
    return {
        path.relative_to(POLICY_ROOT): (
            _read_policy(path) if path in required_paths else _read_future_policy(path)
        )
        for path in paths
    }


def _validate_inventory(documents):
    required_paths = {
        _policy_path(document_id).relative_to(POLICY_ROOT) for document_id in REQUIRED_IDS
    }
    assert required_paths <= documents.keys(), "Missing seed or Batch 1 policy paths"

    ids = set()
    slots = set()
    predecessors = set()
    for path, (fields, _) in documents.items():
        assert set(fields) == REQUIRED_FIELDS, f"Unexpected metadata fields in {path}"
        document_id = _scalar(fields["document_id"])
        family = _scalar(fields["policy_id"])
        language = _scalar(fields["language"])
        version = _scalar(fields["version"])
        assert re.fullmatch(r"[1-9][0-9]*", version) and not fields["version"].lstrip().startswith(
            ("'", '"')
        ), f"Invalid version in {path}"
        slot = (family, language, int(version))

        assert path.parts == (language, f"{document_id}.md"), f"Policy path mismatch: {path}"
        assert slot in AUTHORIZED_SLOTS, f"Unauthorized policy slot: {slot}"
        assert document_id not in ids, f"Duplicate document ID: {document_id}"
        assert slot not in slots, f"Duplicate policy slot: {slot}"
        ids.add(document_id)
        slots.add(slot)

        if document_id in REQUIRED_IDS:
            assert fields["approval_status"] == "approved"
            assert fields["effective_date"] == '"2026-09-01"'
            assert fields["expires_at"] == "null"
        assert _scalar(fields["classification"]) in CLASSIFICATIONS
        _read_roles(fields["allowed_roles"])
        for field in ("title", "department", "approval_status", "source_type", "source_license"):
            _scalar(fields[field])
        for day in (_scalar(fields["effective_date"]), _nullable(fields["expires_at"])):
            if day is not None:
                assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", day), f"Invalid policy date: {day}"
                date.fromisoformat(day)

        predecessor = _nullable(fields["supersedes"])
        assert predecessor != document_id, f"Self-supersession: {document_id}"
        expected = HISTORICAL_IDS.get((family, language, slot[2] - 1))
        assert predecessor == expected, f"Invalid predecessor link for {document_id}: {predecessor!r}"
        if slot in HISTORICAL_IDS or document_id in HISTORICAL_IDS.values():
            assert HISTORICAL_IDS.get(slot) == document_id, f"Unauthorized predecessor ID: {document_id}"
        if predecessor is not None:
            predecessors.add(predecessor)

    assert REQUIRED_IDS <= ids, "Missing seed or Batch 1 document IDs"
    unresolved = predecessors - ids
    assert unresolved <= set(HISTORICAL_IDS.values())
    if slots == AUTHORIZED_SLOTS:
        assert not unresolved, f"Unresolved links in complete inventory: {unresolved}"
    return unresolved


@pytest.mark.parametrize("document_id", BATCH_IDENTITIES)
def test_batch_policy_has_exact_front_matter_and_filename(document_id):
    path = _policy_path(document_id)
    fields, _ = _read_policy(path)
    family, language, version = document_id.rsplit("-", 2)
    title, department, classification, roles, supersedes = BATCH_IDENTITIES[document_id]

    assert len(fields) == 14
    assert fields == {
        "policy_id": family,
        "document_id": document_id,
        "version": version.removeprefix("V"),
        "language": language.lower(),
        "title": title,
        "department": department,
        "classification": classification,
        "allowed_roles": roles,
        "effective_date": '"2026-09-01"',
        "expires_at": "null",
        "supersedes": supersedes,
        "approval_status": "approved",
        "source_type": "synthetic_policy",
        "source_license": "project_owned",
    }
    assert path.name == f"{fields['document_id']}.md"
    assert path.parent.name == fields["language"]


@pytest.mark.parametrize("document_id", BATCH_IDENTITIES)
def test_batch_policy_has_sections_table_approval_and_exception(document_id):
    path = _policy_path(document_id)
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
    assert table is not None
    assert len({line.count("|") for line in table.group().splitlines()}) == 1

    assert re.search(
        r"^\*\*(?:Approbation conditionnelle|Conditional approval)\.\*\* "
        r"(?:Si|If) \S[^\n]*, \S[^\n]*\.$",
        body,
        re.MULTILINE,
    )
    assert re.search(
        r"^\*\*Exception\.\*\* (?:Si|En cas de|If|In a) \S[^\n]*, \S[^\n]*\.$",
        body,
        re.MULTILINE,
    )

    normalized = path.read_text(encoding="utf-8").casefold()
    found = [marker for marker in PROHIBITED_MARKERS if marker.casefold() in normalized]
    assert not found, f"Unfinished markers in {path}: {found}"


def test_corpus_preserves_batch_one_and_uses_only_authorized_slots(corpus):
    _validate_inventory(corpus)


@pytest.mark.parametrize(
    ("document_id", "notice", "duration_rule", "confidentiality", "manager_scope", "emergency", "return_deadline"),
    [
        (
            "LEAVE-001-FR-V2",
            "Pour un congé planifié, le salarié transmet les dates demandées au moins cinq jours ouvrés avant le départ.",
            "Si une demande de congé ordinaire couvre plus de cinq jours ouvrés consécutifs, le responsable hiérarchique confirme la continuité de service et les RH valident le solde avant de notifier l’accord.",
            "Le registre de traitement, les soldes individuels et les justificatifs sont réservés à l’équipe RH.",
            "Le responsable reçoit uniquement les dates et le statut nécessaires à l’organisation du service.",
            "En cas d’urgence familiale, les RH peuvent autoriser une absence sans validation préalable du responsable hiérarchique.",
            "Le salarié régularise sa demande dans les deux jours ouvrés suivant son retour.",
        ),
        (
            "LEAVE-001-EN-V2",
            "For planned leave, the employee submits the requested dates at least five working days before departure.",
            "If a request for ordinary leave covers more than five consecutive working days, the line manager confirms service continuity and HR validates the balance before notifying the employee of approval.",
            "The processing register, individual balances, and supporting documents are reserved for HR.",
            "The line manager receives only the dates and status needed to organize service coverage.",
            "In a family emergency, HR may authorize an absence without prior validation by the line manager.",
            "The employee completes the formal request within two working days after returning.",
        ),
    ],
    ids=["fr", "en"],
)
def test_leave_counterparts_preserve_deadlines_approvals_and_confidentiality(
    document_id, notice, duration_rule, confidentiality, manager_scope, emergency, return_deadline
):
    fields, body = _read_policy(_policy_path(document_id))
    assert fields["classification"] == "RESTRICTED"
    assert fields["allowed_roles"] == "[hr]"
    for clause in (notice, duration_rule, confidentiality, manager_scope, emergency, return_deadline):
        assert clause in body, f"Missing Leave clause in {document_id}: {clause}"


@pytest.mark.parametrize(
    ("document_id", "meal_rule", "approval_rule", "deadline_rule", "receipt_exception"),
    [
        (
            "EXPENSES-001-FR-V2",
            rf"^\| Repas en mission sur le terrain \| Plafond de {AMOUNT} MAD par repas si l’entreprise ne fournit pas le repas dans le cadre de la mission \| Reçu détaillé \|$",
            rf"Si le coût estimé des achats opérationnels dépasse {AMOUNT} MAD, le responsable hiérarchique et Finance doivent approuver l’achat avant tout engagement de la dépense\.$",
            "Le salarié soumet sa demande par le canal interne des dépenses dans les dix jours ouvrés suivant la date de la dépense. Les demandes de remboursement liées aux déplacements professionnels suivent le délai fixé par la politique de déplacements professionnels.",
            "Si le reçu original ne peut être obtenu, Finance peut accepter une copie émise par le fournisseur, accompagnée d’une preuve de paiement et d’une explication de l’absence de l’original.",
        ),
        (
            "EXPENSES-001-EN-V2",
            rf"^\| Meal during a field assignment \| Limit of {AMOUNT} MAD per meal if the company does not provide the meal as part of the assignment \| Itemized receipt \|$",
            rf"If the estimated cost of operational purchases exceeds {AMOUNT} MAD, both the line manager and Finance must approve the purchase before the expense is committed\.$",
            "The employee submits the claim through the internal expenses channel within ten working days of the date of the expense. Business travel claims follow the submission deadline in the Business Travel Policy.",
            "If the original receipt cannot be obtained, Finance may accept a supplier-issued copy with proof of payment and an explanation of why the original is unavailable.",
        ),
        (
            "EXPENSES-001-AR-V2",
            rf"^\| وجبة خلال مهمة ميدانية \| سقف {AMOUNT} درهمًا مغربيًا \(MAD\) لكل وجبة إذا لم توفر الشركة الوجبة ضمن المهمة \| إيصال مفصل \|$",
            rf"إذا تجاوزت التكلفة التقديرية للمشتريات التشغيلية {AMOUNT} درهم مغربي \(MAD\)، وجب الحصول على موافقة المسؤول المباشر ومصلحة المالية قبل الالتزام بالنفقة\.$",
            "يقدّم الموظف الطلب عبر قناة المصاريف الداخلية خلال عشرة أيام عمل من تاريخ المصروف. تخضع مطالبات السفر المهني لمهلة التقديم المحددة في سياسة السفر المهني.",
            "إذا تعذر الحصول على الإيصال الأصلي، يجوز لمصلحة المالية قبول نسخة صادرة عن المورد مع إثبات الدفع وبيان سبب تعذر تقديم الأصل.",
        ),
    ],
    ids=["fr", "en", "ar"],
)
def test_expenses_counterparts_preserve_limits_deadlines_and_receipt_exception(
    document_id, meal_rule, approval_rule, deadline_rule, receipt_exception
):
    fields, body = _read_policy(_policy_path(document_id))
    assert fields["classification"] == "INTERNAL"
    assert fields["allowed_roles"] == ALL_READ_ROLES

    meal = re.search(meal_rule, body, re.MULTILINE)
    approval = re.search(approval_rule, body, re.MULTILINE)
    assert meal is not None, f"Missing conditional meal cap in {document_id}"
    assert approval is not None, f"Missing prior dual approval above threshold in {document_id}"
    assert re.sub(r"[\s,]", "", meal.group(1)) == "150"
    assert re.sub(r"[\s,]", "", approval.group(1)) == "1000"
    assert deadline_rule in body.splitlines()
    assert any(line.endswith(receipt_exception) for line in body.splitlines())


def test_remote_work_preserves_weekly_limit_and_documented_hr_exception():
    _, body = _read_policy(_policy_path("REMOTE-001-FR-V2"))
    assert (
        "**Approbation conditionnelle.** Si les tâches et la couverture du service "
        "le permettent, le responsable hiérarchique peut autoriser au préalable "
        "jusqu’à deux jours de télétravail par semaine."
    ) in body.splitlines()
    assert (
        "**Exception.** Si un besoin temporaire justifie de dépasser deux jours de "
        "télétravail par semaine, le responsable doit approuver la demande et les "
        "RH doivent documenter leur examen avant la mise en œuvre."
    ) in body.splitlines()
    assert "Enregistrer les modalités approuvées et leurs modifications" in body
    assert "Les règles de gestion des données d’Atlas Services continuent de s’appliquer à distance." in body


REQUIRED_RULE_REFERENCES = {
    "Travel planned request": {("TRAVEL-001-FR-V1", 2), ("TRAVEL-001-EN-V1", 2)},
    "Travel dual approval": {("TRAVEL-001-FR-V1", 2), ("TRAVEL-001-EN-V1", 2)},
    "Travel booking and changes": {("TRAVEL-001-FR-V1", 3), ("TRAVEL-001-EN-V1", 3)},
    "Travel transport": {("TRAVEL-001-FR-V1", 4), ("TRAVEL-001-EN-V1", 4)},
    "Travel accommodation": {("TRAVEL-001-FR-V1", 4), ("TRAVEL-001-EN-V1", 4)},
    "Travel claim": {("TRAVEL-001-FR-V1", 5), ("TRAVEL-001-EN-V1", 5)},
    "Travel safety emergency": {("TRAVEL-001-FR-V1", 6), ("TRAVEL-001-EN-V1", 6)},
    "Travel lodging exception": {("TRAVEL-001-FR-V1", 6), ("TRAVEL-001-EN-V1", 6)},
    "Leave planned request": {("LEAVE-001-FR-V2", 2)},
    "Leave processing": {("LEAVE-001-FR-V2", 1), ("LEAVE-001-FR-V2", 2)},
    "Leave duration validation": {("LEAVE-001-FR-V2", 3)},
    "Leave balance and scheduling": {("LEAVE-001-FR-V2", 3)},
    "Leave confidentiality": {("LEAVE-001-FR-V2", 4)},
    "Leave family emergency": {("LEAVE-001-FR-V2", 5)},
    "Leave review": {("LEAVE-001-FR-V2", 5)},
    "Expenses scope and exclusions": {("EXPENSES-001-AR-V2", 1), ("EXPENSES-001-AR-V2", 2)},
    "Expenses field meal": {("EXPENSES-001-AR-V2", 2)},
    "Expenses supporting documents": {("EXPENSES-001-AR-V2", 2)},
    "Expenses ordinary claim": {("EXPENSES-001-AR-V2", 3)},
    "Expenses claim contents": {("EXPENSES-001-AR-V2", 3)},
    "Expenses dual approval": {("EXPENSES-001-AR-V2", 4)},
    "Expenses reimbursement review": {("EXPENSES-001-AR-V2", 4), ("EXPENSES-001-AR-V2", 5)},
    "Expenses receipt exception": {("EXPENSES-001-AR-V2", 6)},
}
QUOTED_HEADING = r'(?:«[^»]+»|“[^”]+”|"[^"]+")'
SOURCE_REFERENCE = re.compile(
    rf"(?P<document_id>\x60[^\x60]+\x60|[^\s,;\x60]+)[ \t]+"
    rf"(?P<sections>§\d+|§§\d+[–-]\d+)[ \t]+"
    rf"(?P<headings>{QUOTED_HEADING}(?:,[ \t]*{QUOTED_HEADING})*)"
)


def _documents_by_id(documents):
    return {_scalar(fields["document_id"]): (fields, body) for fields, body in documents.values()}


def _registry_rows(matrix, section, next_section, columns):
    content = matrix.partition(f"## {section}. ")[2].split(f"\n## {next_section}. ", 1)[0]
    lines = [line.strip() for line in content.splitlines() if line.lstrip().startswith("|")]
    assert all(line.endswith("|") for line in lines), f"Malformed registry row in section {section}"
    rows = [
        tuple(cell.strip() for cell in line.strip("|").split("|"))
        for line in lines
    ]
    assert len(rows) >= 2 and rows[0] == columns, f"Missing or invalid registry table {section}"
    assert all(len(row) == len(columns) and all(row) for row in rows), section
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]), section
    return rows[2:]


def _validate_source_references(source, policies):
    references = set()
    cursor = 0
    for match in SOURCE_REFERENCE.finditer(source):
        separator = source[cursor:match.start()].strip()
        if cursor:
            assert separator in {"and", "et", ",", ";"}, f"Invalid citation separator: {separator!r}"
        else:
            assert not separator, f"Unexpected text before citation: {separator!r}"
        cursor = match.end()

        document_id = match["document_id"].strip("\x60")
        assert document_id in policies, f"Unknown source policy: {document_id}"
        _, body = policies[document_id]
        headings = {
            (int(number), heading)
            for number, heading in re.findall(r"^## (\d+)\. (.+)$", body, re.MULTILINE)
        }
        bounds = [int(number) for number in re.findall(r"\d+", match["sections"])]
        sections = range(bounds[0], bounds[-1] + 1)
        titles = [title[1:-1].strip() for title in re.findall(QUOTED_HEADING, match["headings"])]
        assert len(sections) == len(titles) and titles, f"Mismatched sections in {match[0]!r}"
        for number, title in zip(sections, titles, strict=True):
            assert (number, title) in headings, f"Unknown source heading: {document_id} §{number} {title}"
            references.add((document_id, number))
    assert references and not source[cursor:].strip(), f"Invalid source citation: {source!r}"
    return references


def _validate_rule_registry(matrix, documents):
    policies = _documents_by_id(documents)
    columns = ("Rule", "Established requirement", "Source document and section")
    rules = {}
    for rule, requirement, source in _registry_rows(matrix, "A", "B", columns):
        assert rule not in rules, f"Duplicate registry rule: {rule}"
        references = _validate_source_references(source, policies)
        assert REQUIRED_RULE_REFERENCES.get(rule, set()) <= references, f"Missing source for {rule}"
        rules[rule] = requirement
    assert REQUIRED_RULE_REFERENCES.keys() <= rules.keys(), "Missing established registry rules"
    return rules


def test_rule_matrix_records_established_amounts_deadlines_and_conditions(corpus):
    rules = _validate_rule_registry(MATRIX_PATH.read_text(encoding="utf-8"), corpus)

    expected = {
        "Travel planned request": "At least five working days before planned departure",
        "Travel dual approval": (
            "If the total estimated trip cost exceeds 3,000 MAD, both the line manager "
            "and Finance must approve before any booking."
        ),
        "Travel booking and changes": "Operations books from the approved request and checks change fees.",
        "Travel transport": (
            "Use second-class rail when the schedule permits the assignment, "
            "and economy air travel when rail cannot meet the assignment schedule."
        ),
        "Travel accommodation": "The standard cap is 850 MAD per night, with an itemized invoice.",
        "Travel claim": "Submit within ten working days after return",
        "Travel safety emergency": "Send Finance the supporting document and reason within two working days.",
        "Travel lodging exception": (
            "Outside an emergency, send Finance requests to exceed the lodging cap "
            "before booking, with alternatives considered."
        ),
        "Leave planned request": (
            "For planned leave, the employee submits the requested dates "
            "at least five working days before departure."
        ),
        "Leave processing": "The request is pending until notification.",
        "Leave duration validation": (
            "If ordinary leave exceeds five consecutive working days, the line manager "
            "confirms service continuity and HR validates the balance before notifying approval."
        ),
        "Leave balance and scheduling": "HR asks for revised dates if the balance is insufficient.",
        "Leave confidentiality": (
            "HR alone may access the processing register, individual balances, and supporting documents."
        ),
        "Leave family emergency": "The employee regularizes the request within two working days after return.",
        "Leave review": (
            "The employee may contest a balance or decision with HR leadership, which reviews "
            "the case and records its decision in the HR-only register."
        ),
        "Expenses scope and exclusions": (
            "Employee-paid expenses must support an approved assignment. Exclude personal expenses, "
            "amounts paid directly by the company, and expenses already reimbursed."
        ),
        "Expenses field meal": (
            "150 MAD per meal on a field assignment if the company does not provide the meal."
        ),
        "Expenses supporting documents": "Provide an itemized invoice and proof of payment.",
        "Expenses ordinary claim": (
            "Submit within ten working days of the expense through the internal expenses channel. "
            "Travel claims follow the Travel policy deadline."
        ),
        "Expenses claim contents": "Record each expense date, amount, business purpose, and cost center.",
        "Expenses dual approval": (
            "If estimated operational purchases exceed 1,000 MAD, both the line manager "
            "and Finance must approve before the expense is committed."
        ),
        "Expenses reimbursement review": (
            "Eligible amounts are reimbursed at actual cost within applicable caps after review."
        ),
        "Expenses receipt exception": (
            "If the original receipt cannot be obtained, Finance may accept a supplier-issued copy "
            "with proof of payment and an explanation."
        ),
    }
    for rule, clause in expected.items():
        assert clause in rules[rule], rule


def _validate_identity_registry(matrix, documents):
    policies = _documents_by_id(documents)
    columns = ("Document ID", "Policy ID", "Language", "Version",
               "Classification", "Allowed roles", "Supersedes")
    keys = ("document_id", "policy_id", "language", "version",
            "classification", "allowed_roles", "supersedes")
    registered = set()
    for cells in _registry_rows(matrix, "E", "F", columns):
        row = tuple(cell[1:-1] if cell.startswith("\x60") and cell.endswith("\x60") else cell
                    for cell in cells)
        document_id = row[0]
        assert document_id not in registered, f"Duplicate registry identity: {document_id}"
        assert document_id in policies, f"Registry policy does not exist: {document_id}"
        fields, _ = policies[document_id]
        if document_id in REQUIRED_IDS:
            assert row == tuple(fields[key] for key in keys), f"Registry identity mismatch: {document_id}"
        else:
            assert row[:5] == tuple(_scalar(fields[key]) for key in keys[:5]), (
                f"Registry identity mismatch: {document_id}"
            )
            assert _read_roles(row[5]) == _read_roles(fields["allowed_roles"]), (
                f"Registry ACL mismatch: {document_id}"
            )
            assert _nullable(row[6]) == _nullable(fields["supersedes"]), (
                f"Registry supersession mismatch: {document_id}"
            )
        registered.add(document_id)
    assert REQUIRED_IDS <= registered, "Missing seed or Batch 1 registry identities"


def test_rule_matrix_matches_document_identities_acls_and_version_links(corpus):
    _validate_identity_registry(MATRIX_PATH.read_text(encoding="utf-8"), corpus)


def _synthetic_policy(slot, *, document_id=None, **overrides):
    # Fixture values demonstrate structure without assigning metadata to future policies.
    family, language, version = slot
    document_id = document_id or f"{family}-{language.upper()}-V{version}"
    fields = {
        "policy_id": family,
        "document_id": document_id,
        "version": str(version),
        "language": language,
        "title": "Synthetic test policy",
        "department": "Test department",
        "classification": "INTERNAL",
        "allowed_roles": "[employee]",
        "effective_date": '"2026-10-01"',
        "expires_at": '"2027-01-01"',
        "supersedes": HISTORICAL_IDS.get((family, language, version - 1), "null"),
        "approval_status": "approved",
        "source_type": "synthetic_policy",
        "source_license": "project_owned",
        **overrides,
    }
    return {Path(language) / f"{document_id}.md": (fields, "## 1. Fixture section\n")}


@pytest.fixture
def batch_one_corpus():
    paths = (_policy_path(document_id) for document_id in sorted(REQUIRED_IDS))
    return {path.relative_to(POLICY_ROOT): _read_policy(path) for path in paths}


def test_authorized_active_addition_can_use_block_roles(tmp_path, batch_one_corpus):
    extra = _synthetic_policy(("DATA-001", "en", 2))
    path, (fields, body) = next(iter(extra.items()))
    front_matter = [
        f"{key}: {value}" if key != "allowed_roles" else "allowed_roles:\n  - 'employee'\n  - hr"
        for key, value in fields.items()
    ]
    fixture = tmp_path / path.name
    fixture.write_text("\n".join(["---", *front_matter, "---", body]), encoding="utf-8")
    parsed = _read_future_policy(fixture)

    assert _read_roles(parsed[0]["allowed_roles"]) == {"employee", "hr"}
    assert _validate_inventory(batch_one_corpus | {path: parsed}) == _validate_inventory(
        batch_one_corpus
    )


def test_approved_predecessor_can_resolve_an_existing_link(batch_one_corpus):
    remote_en = ("REMOTE-001", "en", 1)
    predecessor_id = HISTORICAL_IDS[remote_en]
    expanded = batch_one_corpus | _synthetic_policy(("REMOTE-001", "en", 2))
    pending = _validate_inventory(expanded)
    assert pending == _validate_inventory(batch_one_corpus) | {predecessor_id}

    with_history = expanded | _synthetic_policy(remote_en, effective_date='"2025-01-01"')
    assert _validate_inventory(with_history) == pending - {predecessor_id}


def test_inventory_accepts_synthetic_full_active_and_historical_slots(batch_one_corpus):
    expanded = batch_one_corpus.copy()
    for slot in ACTIVE_SLOTS:
        for path, policy in _synthetic_policy(slot).items():
            expanded.setdefault(path, policy)
    assert len(expanded) == 20
    pending = set(HISTORICAL_IDS.values())
    assert _validate_inventory(expanded) == pending

    for slot, predecessor in HISTORICAL_IDS.items():
        expanded.update(_synthetic_policy(slot, effective_date='"2025-01-01"'))
        pending.remove(predecessor)
        assert _validate_inventory(expanded) == pending
    assert len(expanded) == 26


@pytest.mark.parametrize(
    "slot",
    [("UNKNOWN-001", "fr", 1), ("TRAVEL-001", "ar", 1), ("DATA-001", "en", 3),
     ("DATA-001", "en", 1), ("DATA-001", "es", 2)],
    ids=["family", "language-slot", "version", "unapproved-history", "unknown-language"],
)
def test_inventory_rejects_unauthorized_slots(batch_one_corpus, slot):
    with pytest.raises(AssertionError, match="Unauthorized policy slot"):
        _validate_inventory(batch_one_corpus | _synthetic_policy(slot))


@pytest.mark.parametrize("path", [Path("fr/DATA-001-EN-V2.md"), Path("en/wrong.md")])
def test_inventory_rejects_inconsistent_policy_paths(batch_one_corpus, path):
    extra = _synthetic_policy(("DATA-001", "en", 2))
    with pytest.raises(AssertionError, match="Policy path mismatch"):
        _validate_inventory(batch_one_corpus | {path: next(iter(extra.values()))})


def test_inventory_rejects_duplicate_document_ids(batch_one_corpus):
    extra = _synthetic_policy(("DATA-001", "en", 2), document_id="DATA-001-FR-V2")
    with pytest.raises(AssertionError, match="Duplicate document ID"):
        _validate_inventory(batch_one_corpus | extra)


def test_inventory_rejects_duplicate_slots_with_different_ids(batch_one_corpus):
    extra = _synthetic_policy(("DATA-001", "fr", 2), document_id="DUPLICATE-DATA-FR-V2")
    with pytest.raises(AssertionError, match="Duplicate policy slot"):
        _validate_inventory(batch_one_corpus | extra)


@pytest.mark.parametrize(
    ("predecessor", "error"),
    [
        ("REMOTE-001-EN-V2", "Self-supersession"),
        ("REMOTE-001-FR-V1", "Invalid predecessor link"),
        ("DATA-001-FR-V1", "Invalid predecessor link"),
        ("UNKNOWN-001-EN-V1", "Invalid predecessor link"),
        ("null", "Invalid predecessor link"),
        ('"null"', "Invalid predecessor link"),
    ],
)
def test_inventory_rejects_invalid_predecessor_links(batch_one_corpus, predecessor, error):
    extra = _synthetic_policy(("REMOTE-001", "en", 2), supersedes=predecessor)
    with pytest.raises(AssertionError, match=error):
        _validate_inventory(batch_one_corpus | extra)


@pytest.mark.parametrize(
    ("slot", "document_id"),
    [
        (("REMOTE-001", "en", 1), "OTHER-REMOTE-EN-V1"),
        (("DATA-001", "en", 2), "REMOTE-001-EN-V1"),
    ],
)
def test_inventory_rejects_wrong_or_misused_historical_ids(batch_one_corpus, slot, document_id):
    extra = _synthetic_policy(slot, document_id=document_id)
    with pytest.raises(AssertionError, match="Unauthorized predecessor ID"):
        _validate_inventory(batch_one_corpus | extra)


@pytest.mark.parametrize(
    "overrides",
    [
        {"classification": "UNKNOWN"},
        {"allowed_roles": "[]"},
        {"allowed_roles": "[ingestion_admin]"},
        {"allowed_roles": "[employee, employee]"},
        {"effective_date": '"next year"'},
        {"expires_at": '"null"'},
        {"extra_field": "unexpected"},
    ],
    ids=["classification", "empty-roles", "unknown-role", "duplicate-roles",
         "bad-date", "quoted-null", "unknown-field"],
)
def test_inventory_rejects_invalid_future_metadata(batch_one_corpus, overrides):
    slot = ("DATA-001", "en", 2)
    _validate_inventory(batch_one_corpus | _synthetic_policy(slot))
    with pytest.raises(AssertionError):
        _validate_inventory(batch_one_corpus | _synthetic_policy(slot, **overrides))


def test_inventory_rejects_missing_batch_one_document(corpus):
    incomplete = corpus.copy()
    del incomplete[Path("fr/ACCESS-001-FR-V2.md")]
    with pytest.raises(AssertionError, match="Missing seed or Batch 1 policy paths"):
        _validate_inventory(incomplete)


def _add_registry_row(matrix, next_section, row):
    marker = f"\n\n## {next_section}. "
    assert marker in matrix
    return matrix.replace(marker, f"\n{row}{marker}", 1)


@pytest.fixture
def batch_one_matrix(batch_one_corpus):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    rule_columns = ("Rule", "Established requirement", "Source document and section")
    identity_columns = (
        "Document ID", "Policy ID", "Language", "Version",
        "Classification", "Allowed roles", "Supersedes",
    )
    rules = (
        row for row in _registry_rows(matrix, "A", "B", rule_columns)
        if row[0] in REQUIRED_RULE_REFERENCES
    )
    identities = (
        row for row in _registry_rows(matrix, "E", "F", identity_columns)
        if row[0].strip("\x60") in REQUIRED_IDS
    )
    # Keep only the original rows used by the fixture. The live matrix is
    # independently validated against the live corpus above.
    rule_table = (rule_columns, ("---",) * len(rule_columns), *rules)
    identity_table = (identity_columns, ("---",) * len(identity_columns), *identities)
    rule_lines = ("| " + " | ".join(row) + " |" for row in rule_table)
    identity_lines = ("| " + " | ".join(row) + " |" for row in identity_table)
    matrix = "\n".join([
        "## A. Fixture rules", *rule_lines, "", "## B. End of rules", "",
        "## E. Fixture identities", *identity_lines, "", "## F. End of identities",
    ])
    _validate_identity_registry(matrix, batch_one_corpus)
    _validate_rule_registry(matrix, batch_one_corpus)
    return matrix


DATA_FIXTURE_ROW = "| DATA-001-EN-V2 | DATA-001 | en | 2 | INTERNAL | [employee] | null |"


@pytest.mark.parametrize("roles", ["[employee]", ALL_READ_ROLES], ids=["synthetic-acl", "approved-acl"])
def test_registry_accepts_additional_identity_and_rule(batch_one_corpus, batch_one_matrix, roles):
    expanded = batch_one_corpus | _synthetic_policy(("DATA-001", "en", 2), allowed_roles=roles)
    _validate_inventory(expanded)
    row = DATA_FIXTURE_ROW.replace("[employee]", roles)
    matrix = _add_registry_row(batch_one_matrix, "F", row)
    matrix = _add_registry_row(
        matrix, "B",
        "| Fixture rule | Fixture requirement | \x60DATA-001-EN-V2\x60 §1 “Fixture section” |",
    )
    _validate_identity_registry(matrix, expanded)
    assert _validate_rule_registry(matrix, expanded)["Fixture rule"] == "Fixture requirement"


@pytest.mark.parametrize(
    ("row", "replace", "error"),
    [
        ("| DATA-001-EN-V2 | DATA-001 | en | 2 | PUBLIC | [employee] | null |",
         True, "Registry identity mismatch: DATA-001-EN-V2"),
        ("| DATA-001-EN-V2 | DATA-001 | en | 2 | INTERNAL | [hr] | null |",
         True, "Registry ACL mismatch: DATA-001-EN-V2"),
        ("| DATA-001-EN-V2 | DATA-001 | en | 2 | INTERNAL | [employee] | DATA-001-FR-V1 |",
         True, "Registry supersession mismatch: DATA-001-EN-V2"),
        ("| TRAVEL-001-FR-V1 | TRAVEL-001 | fr | 1 | PUBLIC | [employee] | null |",
         False, "Duplicate registry identity: TRAVEL-001-FR-V1"),
        ("| UNKNOWN-001-EN-V2 | UNKNOWN-001 | en | 2 | INTERNAL | [employee] | null |",
         False, "Registry policy does not exist: UNKNOWN-001-EN-V2"),
    ],
    ids=["classification", "acl", "supersession", "duplicate-identity", "unknown-policy"],
)
def test_registry_rejects_inconsistent_identity_rows(batch_one_corpus, batch_one_matrix, row, replace, error):
    expanded = batch_one_corpus | _synthetic_policy(("DATA-001", "en", 2))
    matrix = _add_registry_row(batch_one_matrix, "F", DATA_FIXTURE_ROW)
    _validate_identity_registry(matrix, expanded)
    invalid = matrix.replace(DATA_FIXTURE_ROW, row, 1) if replace else _add_registry_row(matrix, "F", row)
    with pytest.raises(AssertionError, match=error):
        _validate_identity_registry(invalid, expanded)


def test_registry_accepts_original_citations_in_different_order(corpus):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    french = "TRAVEL-001-FR-V1 §2 « Autorisation de la mission »"
    english = "TRAVEL-001-EN-V1 §2 “Trip authorization”"
    reordered = matrix.replace(f"{french} and {english}", f"{english} and {french}", 1)
    assert reordered != matrix
    _validate_rule_registry(reordered, corpus)


def test_registry_requires_original_source_references(corpus):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    _validate_rule_registry(matrix, corpus)
    omitted = matrix.replace("TRAVEL-001-FR-V1 §2 « Autorisation de la mission » and ", "", 1)
    assert omitted != matrix
    with pytest.raises(AssertionError, match="Missing source for Travel planned request"):
        _validate_rule_registry(omitted, corpus)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("UNKNOWN-001-FR-V1 §2 « Autorisation de la mission »", "Unknown source policy"),
        ("TRAVEL-001-FR-V1 §3 « Autorisation de la mission »", "Unknown source heading"),
        ("TRAVEL-001-FR-V1 §2 « Autorisation de la mission » extra text", "Invalid source citation"),
    ],
)
def test_registry_rejects_invalid_new_source_references(batch_one_corpus, batch_one_matrix, source, error):
    row = f"| Fixture rule | Fixture requirement | {source} |"
    matrix = _add_registry_row(batch_one_matrix, "B", row)
    with pytest.raises(AssertionError, match=error):
        _validate_rule_registry(matrix, batch_one_corpus)
