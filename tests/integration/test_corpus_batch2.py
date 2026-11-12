from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

import pytest

from .test_corpus_batch1 import (
    ACTIVE_SLOTS,
    ALL_READ_ROLES,
    HISTORICAL_IDS,
    MATRIX_PATH,
    POLICY_ROOT,
    PROHIBITED_MARKERS,
    REPOSITORY_ROOT,
    REQUIRED_FIELDS,
    _documents_by_id,
    _nullable,
    _policy_path,
    _read_roles,
    _registry_rows,
    _validate_identity_registry,
    _validate_inventory,
    _validate_rule_registry,
    _validate_source_references,
    corpus,
)
from .test_corpus_expansion import ARABIC_WORD, _read_policy


IDENTITIES = {
    # title, department, classification, roles, predecessor
    "REMOTE-001-EN-V2": ("Remote Work Policy", "HR", "INTERNAL", ALL_READ_ROLES, "REMOTE-001-EN-V1"),
    "ACCESS-001-EN-V2": ("Access Control Policy", "IT Security", "RESTRICTED", "[it_security]", "null"),
    "DATA-001-EN-V2": ("Data Handling Policy", "IT Security", "INTERNAL", ALL_READ_ROLES, "null"),
    "INCIDENT-001-EN-V1": ("Incident Reporting Policy", "IT Security", "INTERNAL", ALL_READ_ROLES, "null"),
    "PROCUREMENT-001-EN-V1": ("Procurement Policy", "Finance", "INTERNAL", ALL_READ_ROLES, "null"),
    "REMOTE-001-AR-V2": ("سياسة العمل عن بُعد", "HR", "INTERNAL", ALL_READ_ROLES, "null"),
    "LEAVE-001-AR-V2": ("سياسة الإجازات والغياب", "HR", "RESTRICTED", "[hr]", "null"),
    "DATA-001-AR-V2": ("سياسة التعامل مع البيانات", "IT Security", "INTERNAL", ALL_READ_ROLES, "null"),
}

# Step 52 baseline at 9f71633, with line endings normalized across platforms.
BASELINE_HASHES = {
    "data/policies/ar/EXPENSES-001-AR-V2.md": "33537b2eebfdeb8f80eb8e5dbdad9d5847a5f51d5112650c45c2f0baa599ff97",
    "data/policies/en/EXPENSES-001-EN-V2.md": "e7e6c1ce4fc014d4c86637f5db2034a9a16bbf59bf7139a8cedf44a2c185a55d",
    "data/policies/en/LEAVE-001-EN-V2.md": "796f25909600838da94512f82f4e16a1d1ab9c988e2bb976e60defd07c4cee3c",
    "data/policies/en/TRAVEL-001-EN-V1.md": "c558483aea62580d3dcf9899afdc4ef1b99bdc17fd5352ef5618a38609759917",
    "data/policies/fr/ACCESS-001-FR-V2.md": "596606727277788038ad7592199755cb6a1be9dadaab647a620990c2309d8016",
    "data/policies/fr/DATA-001-FR-V2.md": "87d0d5112113c9ca18b49410bf784d876828e09f6a319b7dbfdfa600a05868be",
    "data/policies/fr/EXPENSES-001-FR-V2.md": "f0b6dda7c3359894b260ffa99191fee8bd3f82502debab1ee317439e6bdebee1",
    "data/policies/fr/INCIDENT-001-FR-V1.md": "abaf26c6f4d430a66e624391b814dc1da86011ee328c42f8021370cf41ca605f",
    "data/policies/fr/LEAVE-001-FR-V2.md": "0ad28f9f8abf2da6a42a864f23b87b494d4e87b172404712a6daab3befac28e7",
    "data/policies/fr/PROCUREMENT-001-FR-V1.md": "a2d8e1d96e0fa6d7b5f1437f830ecee1dc27fd6dfc30c9dab332ea4cc3b66fc3",
    "data/policies/fr/REMOTE-001-FR-V2.md": "19b9ddca80f968d258ce5119363dbd8be2a70706cc73a099b237dbf0f56516c7",
    "data/policies/fr/TRAVEL-001-FR-V1.md": "9669edb6b8017990973835adea2f2637aa846b77cc6fbc68ac39f6cc3ec92f6c",
    "eval/datasets/dev_candidates.jsonl": "dce7427c1e096b5d0cd81f6b928883a0ef2c707f0cfa6dce7826d66f9aff916c",
}

IDENTITY_COLUMNS = (
    "Document ID", "Policy ID", "Language", "Version", "Classification", "Allowed roles", "Supersedes",
)
RULE_COLUMNS = ("Rule", "Established requirement", "Source document and section")
PENDING_LINKS = {
    "REMOTE-001-FR-V2": "REMOTE-001-FR-V1",
    "REMOTE-001-EN-V2": "REMOTE-001-EN-V1",
    "LEAVE-001-FR-V2": "LEAVE-001-FR-V1",
    "EXPENSES-001-FR-V2": "EXPENSES-001-FR-V1",
    "ACCESS-001-FR-V2": "ACCESS-001-FR-V1",
    "DATA-001-FR-V2": "DATA-001-FR-V1",
}


def _expected_metadata(document_id):
    family, language, version = document_id.rsplit("-", 2)
    title, department, classification, roles, supersedes = IDENTITIES[document_id]
    return {
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


def _check_identity(document_id, fields):
    assert fields.keys() == REQUIRED_FIELDS, f"Metadata fields: {document_id}"
    for key, expected in _expected_metadata(document_id).items():
        assert fields[key] == expected, f"Unexpected {key}: {document_id}"


@pytest.mark.parametrize("document_id", IDENTITIES)
def test_batch_two_policy_has_exact_metadata_and_path(document_id):
    path = _policy_path(document_id)
    fields, _ = _read_policy(path)

    _check_identity(document_id, fields)
    assert path.relative_to(POLICY_ROOT).parts == (fields["language"], f"{document_id}.md")


def _sections(body):
    headings = list(re.finditer(r"^## ([1-9][0-9]*)\. (.+)$", body, re.MULTILINE))
    return {
        int(heading[1]): (heading[2], body[heading.end():end].strip())
        for heading, end in zip(headings, [h.start() for h in headings[1:]] + [len(body)])
    }


@pytest.mark.parametrize("document_id", IDENTITIES)
def test_batch_two_policy_has_numbered_sections_table_and_no_placeholders(document_id):
    path = _policy_path(document_id)
    fields, body = _read_policy(path)
    numbers = re.findall(r"^## ([1-9][0-9]*)\. \S", body, re.MULTILINE)
    assert len(numbers) >= 5
    assert list(map(int, numbers)) == list(range(1, len(numbers) + 1))
    assert "Atlas Services" in body

    table = re.search(r"(?:^\|.+\|$\n?){3,}", body, re.MULTILINE)
    assert table, f"Missing table: {document_id}"
    rows = [tuple(cell.strip() for cell in line[1:-1].split("|"))
            for line in table[0].splitlines()]
    assert len(rows[0]) >= 3
    assert all(len(row) == len(rows[0]) and all(row) for row in rows)
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1])

    source = path.read_text(encoding="utf-8").casefold()
    markers = (*PROHIBITED_MARKERS, "all_roles")
    assert not [marker for marker in markers if marker.casefold() in source]
    if fields["language"] == "ar":
        for heading, content in _sections(body).values():
            assert ARABIC_WORD.search(heading)
            assert len(ARABIC_WORD.findall(content)) >= 8
        assert "**الموافقة المشروطة.**" in body
        assert "**استثناء.**" in body
    else:
        assert "**Conditional approval.**" in body
        assert "**Exception.**" in body


def test_corpus_has_all_twenty_active_slots_and_only_authorized_history(corpus):
    pending = _validate_inventory(corpus)
    policies = _documents_by_id(corpus)
    historical = set(HISTORICAL_IDS.values())
    active = {document_id: fields for document_id, (fields, _) in policies.items()
              if document_id not in historical}
    slots = {(fields["policy_id"], fields["language"], int(fields["version"]))
             for fields in active.values()}
    assert slots == ACTIVE_SLOTS
    assert len(active) == 20
    assert Counter(fields["language"] for fields in active.values()) == {"fr": 8, "en": 8, "ar": 4}
    assert {_id: link for _id, fields in active.items()
            if (link := _nullable(fields["supersedes"])) is not None} == PENDING_LINKS
    assert pending == set(PENDING_LINKS.values()) - policies.keys()
    for predecessor in historical & policies.keys():
        assert _nullable(policies[predecessor][0]["supersedes"]) is None


@pytest.mark.parametrize("path", BASELINE_HASHES)
def test_pre_batch_two_policies_and_dev_labels_keep_reviewed_content(path):
    text = (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
    assert sha256(text.encode("utf-8")).hexdigest() == BASELINE_HASHES[path]


def _require_clauses(corpus, document_id, expected):
    path = _policy_path(document_id).relative_to(POLICY_ROOT)
    _, body = corpus[path]
    sections = _sections(body)
    for number, clause in expected:
        assert clause in sections[number][1], f"{document_id} §{number}: {clause}"


REMOTE_RULES = {
    "REMOTE-001-FR-V2": [
        (2, "Si les tâches et la couverture du service le permettent, le responsable hiérarchique peut autoriser au préalable jusqu’à deux jours de télétravail par semaine."),
        (3, "Toute modification de l’organisation approuvée suit la même procédure avant de prendre effet."),
        (4, "Il limite le partage aux personnes autorisées et signale tout incident selon la politique de signalement des incidents."),
        (5, "Si un besoin temporaire justifie de dépasser deux jours de télétravail par semaine, le responsable doit approuver la demande et les RH doivent documenter leur examen avant la mise en œuvre."),
        (5, "En cas d’interruption empêchant de travailler à distance, le salarié informe le responsable, qui convient avec lui d’une solution de remplacement compatible avec le service."),
    ],
    "REMOTE-001-EN-V2": [
        (2, "If the tasks and service coverage allow it, the line manager may authorize up to two remote-work days per week in advance."),
        (2, "The employee waits for approval before working remotely."),
        (3, "| HR | Record approved arrangements and changes | Current arrangements and change date |"),
        (3, "Any change to an approved arrangement follows the same procedure before taking effect."),
        (4, "Employees use approved work tools and approved storage and transfer channels, limit sharing to authorized recipients, and report incidents under the Incident Reporting Policy."),
        (5, "If a temporary need justifies exceeding two remote-work days per week, the line manager must approve the request and HR must document its review before the arrangement is implemented."),
        (5, "If an interruption prevents remote work, the employee informs the line manager and they agree on an alternative that meets service needs."),
    ],
    "REMOTE-001-AR-V2": [
        (2, "إذا سمحت طبيعة المهام ومتطلبات استمرارية العمل، جاز للمسؤول المباشر أن يوافق مسبقًا على العمل عن بُعد لمدة لا تتجاوز يومين في الأسبوع."),
        (2, "ينتظر الموظف الموافقة قبل بدء العمل عن بُعد."),
        (3, "| الموارد البشرية | تسجيل الترتيبات المعتمدة وتعديلاتها | الترتيبات السارية وتاريخ التعديل |"),
        (3, "يخضع أي تعديل في الترتيبات المعتمدة للإجراء نفسه قبل أن يصبح ساريًا."),
        (4, "ويستخدم الموظف أدوات العمل ومساحات التخزين وقنوات النقل المعتمدة، ويقتصر في المشاركة على الأشخاص المخوّلين، ويبلّغ عن أي حادث وفق سياسة الإبلاغ عن الحوادث."),
        (5, "إذا بررت حاجة مؤقتة تجاوز يومين من العمل عن بُعد في الأسبوع، وجب أن يوافق المسؤول المباشر على الطلب وأن توثّق الموارد البشرية مراجعتها قبل تطبيق الترتيب."),
        (5, "وإذا حال انقطاع دون العمل عن بُعد، أبلغ الموظف مسؤوله المباشر واتفق معه على بديل يلائم احتياجات العمل."),
    ],
}

LEAVE_RULES = {
    "LEAVE-001-FR-V2": [
        (2, "Pour un congé planifié, le salarié transmet les dates demandées au moins cinq jours ouvrés avant le départ."),
        (3, "Si une demande de congé ordinaire couvre plus de cinq jours ouvrés consécutifs, le responsable hiérarchique confirme la continuité de service et les RH valident le solde avant de notifier l’accord."),
        (4, "Le registre de traitement, les soldes individuels et les justificatifs sont réservés à l’équipe RH."),
        (4, "Le responsable reçoit uniquement les dates et le statut nécessaires à l’organisation du service."),
        (5, "En cas d’urgence familiale, les RH peuvent autoriser une absence sans validation préalable du responsable hiérarchique."),
        (5, "Le salarié régularise sa demande dans les deux jours ouvrés suivant son retour."),
    ],
    "LEAVE-001-EN-V2": [
        (2, "For planned leave, the employee submits the requested dates at least five working days before departure."),
        (3, "If a request for ordinary leave covers more than five consecutive working days, the line manager confirms service continuity and HR validates the balance before notifying the employee of approval."),
        (4, "The processing register, individual balances, and supporting documents are reserved for HR."),
        (4, "The line manager receives only the dates and status needed to organize service coverage."),
        (5, "In a family emergency, HR may authorize an absence without prior validation by the line manager."),
        (5, "The employee completes the formal request within two working days after returning."),
    ],
    "LEAVE-001-AR-V2": [
        (2, "في حالة الإجازة المخططة، يرسل الموظف التواريخ المطلوبة قبل بدء الإجازة بخمسة أيام عمل على الأقل."),
        (2, "ويظل الطلب قيد النظر حتى صدور هذا الإبلاغ."),
        (3, "إذا تجاوزت الإجازة العادية المطلوبة خمسة أيام عمل متتالية، وجب أن يؤكد المسؤول المباشر استمرارية العمل وأن تتحقق الموارد البشرية من كفاية الرصيد قبل إبلاغ الموظف بالموافقة."),
        (3, "إذا لم يكفِ الرصيد المتاح لتغطية المدة المطلوبة، طلبت الموارد البشرية تواريخ بديلة قبل اتخاذ القرار."),
        (4, "يقتصر الاطلاع على سجل معالجة الطلبات والأرصدة الفردية والمستندات المؤيدة على فريق الموارد البشرية."),
        (4, "ولا يتلقى المسؤول المباشر إلا التواريخ وحالة الطلب اللازمة لتنظيم تغطية العمل."),
        (4, "ولا تُذكر في الجدول المشترك تفاصيل الأسباب الطبية أو العائلية."),
        (5, "في حالة طارئ عائلي، يجوز للموارد البشرية أن تأذن بالغياب دون موافقة مسبقة من المسؤول المباشر."),
        (5, "ويستكمل الموظف إجراءات الطلب خلال يومي عمل من عودته."),
        (5, "وإذا اعترض الموظف على الرصيد أو القرار، أحال الاعتراض إلى إدارة الموارد البشرية، التي تعيد دراسة الملف وتدوّن قرارها في السجل المخصص للموارد البشرية وحدها."),
    ],
}

ACCESS_RULES = {
    "ACCESS-001-FR-V2": [
        (2, "l’accès peut être attribué après validation du responsable hiérarchique et approbation de la Sécurité informatique."),
        (3, "Enregistrer la date de fin et retirer l’accès à l’échéance"),
        (4, "La Sécurité informatique revoit les droits lors d’un changement de poste et aux revues planifiées."),
        (5, "En cas de désaccord sur le besoin, l’attribution reste suspendue pendant le réexamen."),
    ],
    "ACCESS-001-EN-V2": [
        (2, "If the need and scope are justified, access may be granted after line manager validation and IT Security approval."),
        (2, "IT Security asks for a correction before granting access."),
        (3, "| Temporary access | Only the resources and duration strictly necessary | Record the end date and remove access at expiry |"),
        (4, "IT Security reviews permissions when a role changes and at scheduled reviews."),
        (4, "When someone leaves, IT Security revokes access as soon as the departure takes effect, upon notification from the responsible department."),
        (5, "If a temporary need is urgent, IT Security may prioritize the request after line manager validation."),
        (5, "IT Security approval is still required before any access is granted."),
        (5, "IT Security documents the justification, its approval, the scope, and the end date."),
        (5, "If the need is disputed, the requested access grant remains on hold until the dispute is resolved."),
    ],
}

DATA_RULES = {
    "DATA-001-FR-V2": [
        (1, "Chaque salarié vérifie la classification, le besoin professionnel et l’autorisation du destinataire avant de partager des données."),
        (2, "La classification ne confère aucun accès par elle-même."),
        (2, "Si l’autorisation du destinataire est inconnue, le salarié suspend le partage et demande confirmation au propriétaire des données."),
        (3, "Si un transfert vers un destinataire externe est nécessaire, le propriétaire des données et la Sécurité informatique doivent approuver le transfert avant l’envoi."),
        (4, "Il indique le canal, le moment et la catégorie des données concernées, en évitant toute rediffusion des données."),
        (5, "Si le canal habituel est indisponible, le propriétaire des données et la Sécurité informatique peuvent approuver l’utilisation temporaire d’un autre canal autorisé avant tout transfert."),
    ],
    "DATA-001-EN-V2": [
        (1, "Before sharing, employees verify the classification, the business need, and the recipient's authorization."),
        (2, "| RESTRICTED | Explicit authorization of the recipient for the document | Use the authorized space and share only what is strictly necessary |"),
        (2, "Classification alone confers no access."),
        (2, "If recipient authorization is unknown, the employee suspends sharing and seeks confirmation from the data owner."),
        (3, "Employees use company-approved storage spaces and transfer channels for the data concerned."),
        (3, "If transfer to an external recipient is necessary, the data owner and IT Security must approve the transfer before it is sent."),
        (4, "Employees promptly report any suspected disclosure to IT Security under the Incident Reporting Policy."),
        (4, "They specify the channel, time, and category of the affected data without redistributing it."),
        (5, "If the usual channel is unavailable, the data owner and IT Security may approve temporary use of another authorized channel before any transfer."),
        (5, "The decision records the justification, duration, and selected channel."),
        (5, "Recipient restrictions continue to apply. If no authorized channel is available, the employee suspends the transfer and contacts IT Security."),
    ],
    "DATA-001-AR-V2": [
        (1, "وقبل مشاركة البيانات، يتحقق الموظف من تصنيفها والحاجة إليها لأغراض العمل ومن صلاحية المستلم للاطلاع عليها."),
        (2, "| RESTRICTED (مقيّد) | تخويل المستلم صراحةً بالاطلاع على الوثيقة | استخدام المساحة المصرّح بها وقصر المشاركة على القدر الضروري تمامًا |"),
        (2, "لا يمنح التصنيف وحده حق الوصول."),
        (2, "وإذا لم يكن تخويل المستلم معلومًا، يوقف الموظف المشاركة ويطلب التأكيد من مالك البيانات."),
        (3, "يستخدم الموظف مساحات التخزين وقنوات النقل التي اعتمدتها الشركة للبيانات المعنية."),
        (3, "إذا دعت الحاجة إلى نقل بيانات إلى مستلم خارجي، وجب الحصول على موافقة مالك البيانات وأمن المعلومات قبل الإرسال."),
        (4, "يبلّغ الموظف أمن المعلومات سريعًا عن أي إفصاح مشتبه به وفق سياسة الإبلاغ عن الحوادث."),
        (4, "ويحدد القناة والوقت وفئة البيانات المعنية دون إعادة نشرها."),
        (5, "إذا تعذّر استخدام القناة المعتادة، جاز لمالك البيانات وأمن المعلومات الموافقة على استخدام قناة أخرى مصرّح بها بصورة مؤقتة قبل أي نقل."),
        (5, "يوثّق القرار المبرر والمدة والقناة المختارة."),
        (5, "وتظل القيود المتعلقة بالمستلمين سارية. وإذا لم تتوفر قناة مصرّح بها، يوقف الموظف النقل ويراجع أمن المعلومات."),
    ],
}

INCIDENT_RULES = {
    "INCIDENT-001-FR-V1": [
        (1, "Le salarié signale une situation suspecte dès qu’il la constate, même si l’incident reste à confirmer."),
        (2, "Le salarié utilise le canal interne désigné de la Sécurité informatique."),
        (3, "La Sécurité informatique reçoit le signalement, apprécie l’impact et l’urgence"),
        (4, "Le salarié préserve les éléments originaux disponibles"),
        (4, "Si une remise en service risque de modifier un élément de preuve, le salarié doit obtenir l’accord de la Sécurité informatique avant d’intervenir."),
        (5, "Si le canal désigné est indisponible, le salarié avertit son responsable par un autre canal interne approuvé afin qu’il relaie rapidement le signalement à la Sécurité informatique."),
    ],
    "INCIDENT-001-EN-V1": [
        (1, "Employees report suspicious situations as soon as they notice them, even if the incident is still unconfirmed."),
        (2, "Employees use the designated internal IT Security channel."),
        (3, "IT Security receives the report, assesses impact and urgency, and tells the reporter what to do next."),
        (4, "Employees preserve available original evidence and use approved channels to send it to IT Security."),
        (4, "If restoring service could alter evidence, the employee must obtain IT Security approval before acting."),
        (5, "If the designated channel is unavailable, the employee alerts their line manager through another approved internal channel so the manager can relay the report promptly to IT Security."),
        (5, "The employee reports any change that increases impact or urgency through the available approved channel."),
    ],
}

PROCUREMENT_RULES = {
    "PROCUREMENT-001-FR-V1": [
        (2, "Si le budget est autorisé et le dossier fournisseur complet, Finance peut approuver le bon de commande avant tout engagement auprès du fournisseur."),
        (2, "Une modification du prix ou du périmètre est soumise au responsable budgétaire et à Finance avant d’être confirmée."),
        (3, "Le demandeur ne valide pas sa propre demande. La personne qui transmet la commande est distincte de celle qui l’approuve."),
        (4, "Le bon de commande ne dispense pas de ces autorisations. Un même poste ne peut pas être réglé par les deux voies."),
        (5, "Le bon de commande approuvé et les autorisations requises restent nécessaires."),
    ],
    "PROCUREMENT-001-EN-V1": [
        (2, "The budget owner verifies the need and budget availability and records their approval."),
        (2, "The requester provides the quote, supplier terms, and reasons for selecting the supplier for review before any commitment."),
        (2, "If the budget is authorized and the supplier documentation is complete, Finance may approve the purchase order before any commitment to the supplier."),
        (2, "Any change to price or scope goes back to the budget owner and Finance for approval before confirmation."),
        (3, "The requester does not approve their own request. The person who sends the order is different from the person who approves it."),
        (3, "Receipt is recorded separately to reconcile the purchase order, delivery, and invoice."),
        (4, "For employee-paid lines, obtain the applicable prior approvals under the Professional Expenses Policy before committing the expense, then submit the claim with supporting documents."),
        (4, "The purchase order does not waive those approvals. A line item may be paid through only one route."),
        (5, "If a need is urgent, the budget owner and Finance may prioritize review before commitment, with a written explanation of the urgency."),
        (5, "The approved purchase order and required approvals still apply."),
    ],
}

SOURCE_CLAUSES = REMOTE_RULES | LEAVE_RULES | ACCESS_RULES | DATA_RULES | INCIDENT_RULES | PROCUREMENT_RULES


@pytest.mark.parametrize("document_id", SOURCE_CLAUSES)
def test_language_counterpart_preserves_selected_source_clauses(corpus, document_id):
    _require_clauses(corpus, document_id, SOURCE_CLAUSES[document_id])


@pytest.mark.parametrize("document_id", [*LEAVE_RULES, *ACCESS_RULES])
def test_restricted_counterparts_keep_same_reader_boundary(corpus, document_id):
    fields, _ = corpus[_policy_path(document_id).relative_to(POLICY_ROOT)]
    assert fields["classification"] == "RESTRICTED"
    assert _read_roles(fields["allowed_roles"]) == (
        {"hr"} if document_id in LEAVE_RULES else {"it_security"}
    )


def test_employee_readable_batch_two_policies_exclude_restricted_procedure_clauses(corpus):
    signatures = {
        "en": (
            "The processing register, individual balances, and supporting documents",
            "In a family emergency, HR may authorize an absence",
            "IT Security documents the justification, its approval, the scope, and the end date",
        ),
        "ar": (
            "سجل معالجة الطلبات والأرصدة الفردية والمستندات المؤيدة",
            "يجوز للموارد البشرية أن تأذن بالغياب",
        ),
    }
    for document_id in IDENTITIES:
        fields, body = corpus[_policy_path(document_id).relative_to(POLICY_ROOT)]
        if "employee" in _read_roles(fields["allowed_roles"]):
            assert all(signature not in body for signature in signatures[fields["language"]]), document_id


def test_matrix_has_all_twenty_active_identities_and_exact_acls(corpus):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    _validate_identity_registry(matrix, corpus)
    rows = _registry_rows(matrix, "E", "F", IDENTITY_COLUMNS)
    registered = {row[0].strip("\x60") for row in rows}
    active_ids = _documents_by_id(corpus).keys() - set(HISTORICAL_IDS.values())
    assert active_ids <= registered
    assert len(active_ids) == 20


BATCH_TWO_REFERENCES = {
    "Remote Work approval": {
        ("REMOTE-001-FR-V2", 2),
        ("REMOTE-001-EN-V2", 2),
        ("REMOTE-001-AR-V2", 2),
    },
    "Remote Work records": {
        ("REMOTE-001-FR-V2", 3),
        ("REMOTE-001-EN-V2", 3),
        ("REMOTE-001-AR-V2", 3),
    },
    "Remote Work data protection": {
        ("REMOTE-001-FR-V2", 4),
        ("REMOTE-001-EN-V2", 4),
        ("REMOTE-001-AR-V2", 4),
    },
    "Remote Work exception and interruption": {
        ("REMOTE-001-FR-V2", 5),
        ("REMOTE-001-EN-V2", 5),
        ("REMOTE-001-AR-V2", 5),
    },
    "Access Control approval": {
        ("ACCESS-001-FR-V2", 2),
        ("ACCESS-001-EN-V2", 2),
    },
    "Access Control least privilege and expiry": {
        ("ACCESS-001-FR-V2", 3),
        ("ACCESS-001-EN-V2", 3),
    },
    "Access Control review and revocation": {
        ("ACCESS-001-FR-V2", 4),
        ("ACCESS-001-EN-V2", 4),
    },
    "Access Control urgent or disputed requests": {
        ("ACCESS-001-FR-V2", 5),
        ("ACCESS-001-EN-V2", 5),
    },
    "Data Handling classification and sharing": {
        ("DATA-001-FR-V2", 2),
        ("DATA-001-EN-V2", 2),
        ("DATA-001-AR-V2", 2),
    },
    "Data Handling storage and external transfers": {
        ("DATA-001-FR-V2", 3),
        ("DATA-001-EN-V2", 3),
        ("DATA-001-AR-V2", 3),
    },
    "Data Handling accidental disclosure": {
        ("DATA-001-FR-V2", 4),
        ("DATA-001-EN-V2", 4),
        ("DATA-001-AR-V2", 4),
    },
    "Data Handling channel exception": {
        ("DATA-001-FR-V2", 5),
        ("DATA-001-EN-V2", 5),
        ("DATA-001-AR-V2", 5),
    },
    "Incident Reporting channel": {
        ("INCIDENT-001-FR-V1", 1), ("INCIDENT-001-FR-V1", 2),
        ("INCIDENT-001-EN-V1", 1), ("INCIDENT-001-EN-V1", 2),
    },
    "Incident Reporting triage and evidence": {
        ("INCIDENT-001-FR-V1", 3), ("INCIDENT-001-FR-V1", 4),
        ("INCIDENT-001-EN-V1", 3), ("INCIDENT-001-EN-V1", 4),
    },
    "Incident Reporting channel exception": {
        ("INCIDENT-001-FR-V1", 5),
        ("INCIDENT-001-EN-V1", 5),
    },
    "Procurement commitment and reapproval": {
        ("PROCUREMENT-001-FR-V1", 2),
        ("PROCUREMENT-001-EN-V1", 2),
    },
    "Procurement separate duties": {
        ("PROCUREMENT-001-FR-V1", 3),
        ("PROCUREMENT-001-EN-V1", 3),
    },
    "Procurement mixed payments": {
        ("PROCUREMENT-001-FR-V1", 4),
        ("PROCUREMENT-001-EN-V1", 4),
    },
    "Procurement urgency": {
        ("PROCUREMENT-001-FR-V1", 5),
        ("PROCUREMENT-001-EN-V1", 5),
    },
    "Leave planned request across languages": {
        ("LEAVE-001-FR-V2", 2),
        ("LEAVE-001-EN-V2", 2),
        ("LEAVE-001-AR-V2", 2),
    },
    "Leave processing across languages": {
        ("LEAVE-001-FR-V2", 1), ("LEAVE-001-FR-V2", 2),
        ("LEAVE-001-EN-V2", 1), ("LEAVE-001-EN-V2", 2),
        ("LEAVE-001-AR-V2", 1), ("LEAVE-001-AR-V2", 2),
    },
    "Leave duration validation across languages": {
        ("LEAVE-001-FR-V2", 3),
        ("LEAVE-001-EN-V2", 3),
        ("LEAVE-001-AR-V2", 3),
    },
    "Leave balance and scheduling across languages": {
        ("LEAVE-001-FR-V2", 3),
        ("LEAVE-001-EN-V2", 3),
        ("LEAVE-001-AR-V2", 3),
    },
    "Leave confidentiality across languages": {
        ("LEAVE-001-FR-V2", 4),
        ("LEAVE-001-EN-V2", 4),
        ("LEAVE-001-AR-V2", 4),
    },
    "Leave family emergency across languages": {
        ("LEAVE-001-FR-V2", 5),
        ("LEAVE-001-EN-V2", 5),
        ("LEAVE-001-AR-V2", 5),
    },
    "Leave review across languages": {
        ("LEAVE-001-FR-V2", 5),
        ("LEAVE-001-EN-V2", 5),
        ("LEAVE-001-AR-V2", 5),
    },
}


def test_matrix_cites_batch_two_rules_and_their_counterparts(corpus):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    _validate_rule_registry(matrix, corpus)
    rows = {rule: source for rule, _, source in _registry_rows(matrix, "A", "B", RULE_COLUMNS)}
    policies = _documents_by_id(corpus)
    for rule, references in BATCH_TWO_REFERENCES.items():
        assert references <= _validate_source_references(rows[rule], policies), rule


def _change_metadata(corpus, document_id, field, value):
    path = _policy_path(document_id).relative_to(POLICY_ROOT)
    fields, body = corpus[path]
    fields = fields.copy()
    if value is None:
        del fields[field]
    else:
        fields[field] = value
    return corpus | {path: (fields, body)}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("classification", "PUBLIC", "Unexpected classification"),
        ("allowed_roles", "[it_security, employee]", "Unexpected allowed_roles"),
        ("allowed_roles", None, "Metadata fields"),
        ("allowed_roles", "[]", "Unexpected allowed_roles"),
    ],
    ids=["wrong-classification", "extra-reader", "missing-roles", "empty-roles"],
)
def test_exact_identity_rejects_acl_changes(corpus, field, value, error):
    document_id = "ACCESS-001-EN-V2"
    path = _policy_path(document_id).relative_to(POLICY_ROOT)
    _check_identity(document_id, corpus[path][0])
    changed = _change_metadata(corpus, document_id, field, value)
    with pytest.raises(AssertionError, match=error):
        _check_identity(document_id, changed[path][0])


@pytest.mark.parametrize(
    ("predecessor", "error"),
    [
        ("DATA-001-EN-V1", "Invalid predecessor link"),
        ("REMOTE-001-FR-V1", "Invalid predecessor link"),
    ],
    ids=["wrong-family", "cross-language"],
)
def test_inventory_rejects_wrong_remote_english_predecessor(corpus, predecessor, error):
    _validate_inventory(corpus)
    changed = _change_metadata(corpus, "REMOTE-001-EN-V2", "supersedes", predecessor)
    with pytest.raises(AssertionError, match=error):
        _validate_inventory(changed)


@pytest.mark.parametrize(
    ("path", "overrides", "error"),
    [
        ("en/UNKNOWN-001-EN-V2.md", {"policy_id": "UNKNOWN-001", "document_id": "UNKNOWN-001-EN-V2"},
         "Unauthorized policy slot"),
        ("ar/DATA-001-EN-V2.md", {"language": "ar"}, "Duplicate document ID"),
        ("en/DUPLICATE-DATA-EN-V2.md", {"document_id": "DUPLICATE-DATA-EN-V2"}, "Duplicate policy slot"),
    ],
    ids=["unknown-slot", "duplicate-id", "duplicate-slot"],
)
def test_inventory_rejects_invalid_additions_after_batch_two(corpus, path, overrides, error):
    _validate_inventory(corpus)
    fields, body = corpus[Path("en/DATA-001-EN-V2.md")]
    invalid = corpus | {Path(path): (fields | overrides, body)}
    with pytest.raises(AssertionError, match=error):
        _validate_inventory(invalid)


@pytest.mark.parametrize(
    ("document_id", "before", "after"),
    [
        ("REMOTE-001-EN-V2", "up to two remote-work days", "up to three remote-work days"),
        ("LEAVE-001-AR-V2", "بخمسة أيام عمل على الأقل", "بأربعة أيام عمل على الأقل"),
    ],
    ids=["remote-weekly-limit", "leave-notice-deadline"],
)
def test_clause_checks_detect_changed_limits(corpus, document_id, before, after):
    path = _policy_path(document_id).relative_to(POLICY_ROOT)
    fields, body = corpus[path]
    expected = SOURCE_CLAUSES[document_id]
    _require_clauses(corpus, document_id, expected)
    assert before in body
    mutated = corpus | {path: (fields, body.replace(before, after, 1))}
    with pytest.raises(AssertionError, match=re.escape(document_id + " §2:")):
        _require_clauses(mutated, document_id, expected)


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ("\x60INTERNAL\x60", "\x60PUBLIC\x60", "Registry identity mismatch: DATA-001-EN-V2"),
        ("\x60[employee, hr, finance, it_security]\x60", "\x60[employee]\x60",
         "Registry ACL mismatch: DATA-001-EN-V2"),
    ],
    ids=["classification", "acl"],
)
def test_matrix_rejects_drift_in_real_english_data_identity(corpus, before, after, error):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    _validate_identity_registry(matrix, corpus)
    row = next(line for line in matrix.splitlines() if line.startswith("|\x20\x60DATA-001-EN-V2\x60 |"))
    assert before in row
    invalid = matrix.replace(row, row.replace(before, after, 1), 1)
    with pytest.raises(AssertionError, match=error):
        _validate_identity_registry(invalid, corpus)
