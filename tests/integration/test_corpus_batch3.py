from collections import Counter
from datetime import date
from hashlib import sha256
import json
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
    REQUIRED_RULE_REFERENCES,
    _documents_by_id,
    _nullable,
    _policy_path,
    _read_roles,
    _registry_rows,
    _validate_identity_registry,
    _validate_inventory,
    _validate_source_references,
    corpus,
)
from .test_corpus_batch2 import (
    BASELINE_HASHES,
    BATCH_TWO_REFERENCES,
    IDENTITY_COLUMNS,
    PENDING_LINKS as RESOLVED_LINKS,
    RULE_COLUMNS,
    _sections,
)
from .test_corpus_expansion import _parse_policy, _read_policy


HISTORICAL_METADATA = {
    # title, department, classification, explicit roles
    "REMOTE-001-FR-V1": ("Politique de télétravail — Version 1", "HR", "INTERNAL", ALL_READ_ROLES),
    "REMOTE-001-EN-V1": ("Remote Work Policy — Version 1", "HR", "INTERNAL", ALL_READ_ROLES),
    "LEAVE-001-FR-V1": ("Politique des congés et absences — Version 1", "HR", "RESTRICTED", "[hr]"),
    "EXPENSES-001-FR-V1": ("Politique des dépenses professionnelles — Version 1", "Finance", "INTERNAL", ALL_READ_ROLES),
    "ACCESS-001-FR-V1": ("Politique de contrôle d'accès — Version 1", "IT Security", "RESTRICTED", "[it_security]"),
    "DATA-001-FR-V1": ("Politique de gestion des données — Version 1", "IT Security", "INTERNAL", ALL_READ_ROLES),
}
VALIDITY_TIME = date(2026, 10, 1)
DEV_DATASET = "eval/datasets/dev_candidates.jsonl"

# Baseline af25189. Text hashes normalize line endings across checkouts.
ACTIVE_HASHES = {
    "DATA-001-AR-V2": "ae8ee21b18100e4368e137dec4e22c9b98bf43928419ad04e4ddb60800eaceb1",
    "EXPENSES-001-AR-V2": "33537b2eebfdeb8f80eb8e5dbdad9d5847a5f51d5112650c45c2f0baa599ff97",
    "LEAVE-001-AR-V2": "8a1b76167587e204b0890bb7f88898621b09e9ff95717284715c2a24e85e25f8",
    "REMOTE-001-AR-V2": "fcc525aefeef649ae4fd935967603c5ac08cba5420c9407a3f05ce5dc3006c43",
    "ACCESS-001-EN-V2": "0577cbeec676a6759a69399290a61e925a55537dda199ab8c13c8fafe6f36ff2",
    "DATA-001-EN-V2": "dfb25cff6cae884023b892cd0e47162a5f709914acb120d6af267e47f2f29227",
    "EXPENSES-001-EN-V2": "e7e6c1ce4fc014d4c86637f5db2034a9a16bbf59bf7139a8cedf44a2c185a55d",
    "INCIDENT-001-EN-V1": "0de9e8ee1c1adc0ce9a21fb63019d7048c77b062f85b0766cc582387a9ffdf47",
    "LEAVE-001-EN-V2": "796f25909600838da94512f82f4e16a1d1ab9c988e2bb976e60defd07c4cee3c",
    "PROCUREMENT-001-EN-V1": "4929c9b612744fb6fe96050bed5ec3c497fd77d46aad830452ce20abc2a3f02e",
    "REMOTE-001-EN-V2": "2062dfd6b0fba7ad7c400136174a1cd4837213a71d529e9b0c2cc7e24b27a324",
    "TRAVEL-001-EN-V1": "c558483aea62580d3dcf9899afdc4ef1b99bdc17fd5352ef5618a38609759917",
    "ACCESS-001-FR-V2": "596606727277788038ad7592199755cb6a1be9dadaab647a620990c2309d8016",
    "DATA-001-FR-V2": "87d0d5112113c9ca18b49410bf784d876828e09f6a319b7dbfdfa600a05868be",
    "EXPENSES-001-FR-V2": "f0b6dda7c3359894b260ffa99191fee8bd3f82502debab1ee317439e6bdebee1",
    "INCIDENT-001-FR-V1": "abaf26c6f4d430a66e624391b814dc1da86011ee328c42f8021370cf41ca605f",
    "LEAVE-001-FR-V2": "0ad28f9f8abf2da6a42a864f23b87b494d4e87b172404712a6daab3befac28e7",
    "PROCUREMENT-001-FR-V1": "a2d8e1d96e0fa6d7b5f1437f830ecee1dc27fd6dfc30c9dab332ea4cc3b66fc3",
    "REMOTE-001-FR-V2": "19b9ddca80f968d258ce5119363dbd8be2a70706cc73a099b237dbf0f56516c7",
    "TRAVEL-001-FR-V1": "9669edb6b8017990973835adea2f2637aa846b77cc6fbc68ac39f6cc3ec92f6c",
}
CURRENT_RULES_HASH = "bb18b632b5a0bde797f457cdb6950840b2a8c086dfe65d2cb5a7388fb89bbb25"


def _check_historical_metadata(document_id, fields):
    family, language, _ = document_id.rsplit("-", 2)
    title, department, classification, roles = HISTORICAL_METADATA[document_id]
    expected = {
        "policy_id": family,
        "document_id": document_id,
        "version": "1",
        "language": language.lower(),
        "title": title,
        "department": department,
        "classification": classification,
        "allowed_roles": roles,
        "effective_date": '"2026-01-01"',
        "expires_at": "null",
        "supersedes": "null",
        "approval_status": "approved",
        "source_type": "synthetic_policy",
        "source_license": "project_owned",
    }
    assert fields.keys() == REQUIRED_FIELDS, f"Unexpected metadata fields: {document_id}"
    for key, value in expected.items():
        assert fields[key] == value, f"Unexpected {key}: {document_id}"


@pytest.mark.parametrize("document_id", HISTORICAL_METADATA)
def test_historical_policy_has_exact_metadata_and_path(document_id):
    path = _policy_path(document_id)
    fields, _ = _read_policy(path)
    _check_historical_metadata(document_id, fields)
    assert path.relative_to(POLICY_ROOT).parts == (fields["language"], f"{document_id}.md")
    if fields["classification"] == "RESTRICTED":
        successor = _policy_path(document_id.removesuffix("V1") + "V2")
        active, _ = _read_policy(successor)
        assert active["classification"] == "RESTRICTED"
        assert _read_roles(active["allowed_roles"]) == _read_roles(fields["allowed_roles"])


@pytest.fixture
def policies(corpus):
    return _documents_by_id(corpus)


def _validate_supersession(policies):
    links = {
        document_id: predecessor
        for document_id, (fields, _) in policies.items()
        if (predecessor := _nullable(fields["supersedes"])) is not None
    }
    for successor, predecessor in links.items():
        assert successor != predecessor, f"Self-supersession: {successor}"
        assert predecessor in policies, f"Missing predecessor: {predecessor}"

    for start in links:
        seen = set()
        cursor = start
        while cursor in links:
            assert cursor not in seen, f"Supersession cycle through {cursor}"
            seen.add(cursor)
            cursor = links[cursor]

    for successor, predecessor in links.items():
        newer, _ = policies[successor]
        older, _ = policies[predecessor]
        assert newer["policy_id"] == older["policy_id"], f"Wrong-family predecessor: {successor}"
        assert newer["language"] == older["language"], f"Wrong-language predecessor: {successor}"
        assert int(older["version"]) < int(newer["version"]), f"Predecessor version must be earlier: {successor}"
        assert date.fromisoformat(_nullable(older["effective_date"])) < date.fromisoformat(
            _nullable(newer["effective_date"])
        ), f"Predecessor effective date must be earlier: {successor}"
        assert older["approval_status"] == "approved", f"Predecessor approval: {predecessor}"
        assert older["supersedes"] == "null", f"V1 predecessor must terminate the link: {predecessor}"

    assert links == RESOLVED_LINKS
    return links


def test_complete_inventory_has_exact_slots_and_resolved_links(corpus, policies):
    assert _validate_inventory(corpus) == set()
    links = _validate_supersession(policies)
    active_ids = policies.keys() - set(links.values())
    assert active_ids == ACTIVE_HASHES.keys()
    assert set(links.values()) == HISTORICAL_METADATA.keys()
    assert len(corpus) == len(policies) == 26
    assert len(active_ids) == 20
    assert len(links) == 6
    assert Counter(policies[key][0]["language"] for key in active_ids) == {"fr": 8, "en": 8, "ar": 4}
    assert Counter(policies[key][0]["language"] for key in links.values()) == {"fr": 5, "en": 1}

    slots = {(fields["policy_id"], fields["language"], int(fields["version"]))
             for fields, _ in policies.values()}
    assert slots == ACTIVE_SLOTS | HISTORICAL_IDS.keys()
    assert len(slots) == len(policies)


def _approved_and_effective(fields, as_of):
    effective = date.fromisoformat(_nullable(fields["effective_date"]))
    return fields["approval_status"] == "approved" and effective <= as_of


def _unexpired(fields, as_of):
    expires = _nullable(fields["expires_at"])
    return expires is None or as_of < date.fromisoformat(expires)


def _current_policy_ids(policies):
    """Model the Phase 2 contract at the fixed validity time."""
    effective = {
        document_id: fields
        for document_id, (fields, _) in policies.items()
        if _approved_and_effective(fields, VALIDITY_TIME)
    }
    superseded = {_nullable(fields["supersedes"]) for fields in effective.values()}
    return {document_id for document_id, fields in effective.items()
            if document_id not in superseded and _unexpired(fields, VALIDITY_TIME)}


def _assert_current_set(eligible):
    historical = eligible & HISTORICAL_METADATA.keys()
    assert not historical, f"Superseded policy selected: {sorted(historical)}"
    assert eligible == ACTIVE_HASHES.keys()


def test_fixed_time_current_policy_set_contains_all_twenty_active_ids(policies):
    _validate_supersession(policies)
    _assert_current_set(_current_policy_ids(policies))


@pytest.mark.parametrize("document_id", HISTORICAL_METADATA)
def test_superseded_policy_is_otherwise_approved_effective_and_unexpired(policies, document_id):
    fields, _ = policies[document_id]
    assert _approved_and_effective(fields, VALIDITY_TIME)
    assert fields["expires_at"] == "null" and _unexpired(fields, VALIDITY_TIME)
    assert document_id not in _current_policy_ids(policies)


def test_validity_checks_alone_would_incorrectly_select_historical_policies(policies):
    _assert_current_set(_current_policy_ids(policies))
    without_supersession = {
        document_id
        for document_id, (fields, _) in policies.items()
        if _approved_and_effective(fields, VALIDITY_TIME) and _unexpired(fields, VALIDITY_TIME)
    }
    assert without_supersession == policies.keys()
    with pytest.raises(AssertionError, match="Superseded policy selected"):
        _assert_current_set(without_supersession)


def _replace_metadata(records, key, **overrides):
    fields, body = records[key]
    return records | {key: (fields | overrides, body)}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_status", "not_approved"),
        ("effective_date", '"2026-10-02"'),
        ("expires_at", '"2026-09-30"'),
    ],
    ids=["unapproved", "not-effective", "expired"],
)
def test_other_validity_conditions_also_exclude_policies(policies, field, value):
    active = "TRAVEL-001-FR-V1"
    _assert_current_set(_current_policy_ids(policies))
    changed = _replace_metadata(policies, active, **{field: value})
    assert _current_policy_ids(changed) == ACTIVE_HASHES.keys() - {active}


def test_supersession_rejects_missing_predecessor(policies):
    _validate_supersession(policies)
    predecessor = "REMOTE-001-EN-V1"
    incomplete = policies.copy()
    del incomplete[predecessor]
    with pytest.raises(AssertionError, match=f"Missing predecessor: {predecessor}"):
        _validate_supersession(incomplete)


@pytest.mark.parametrize(
    ("predecessor", "error"),
    [
        ("REMOTE-001-EN-V1", "Wrong-language predecessor"),
        ("DATA-001-FR-V1", "Wrong-family predecessor"),
        ("REMOTE-001-FR-V2", "Self-supersession"),
    ],
    ids=["wrong-language", "wrong-family", "self-reference"],
)
def test_supersession_rejects_invalid_target(policies, predecessor, error):
    _validate_supersession(policies)
    changed = _replace_metadata(policies, "REMOTE-001-FR-V2", supersedes=predecessor)
    with pytest.raises(AssertionError, match=error):
        _validate_supersession(changed)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("version", "2", "Predecessor version must be earlier"),
        ("effective_date", '"2026-09-01"', "Predecessor effective date must be earlier"),
        ("approval_status", "not_approved", "Predecessor approval"),
        ("supersedes", "REMOTE-001-FR-V2", "Supersession cycle"),
    ],
    ids=["same-version", "same-effective-date", "unapproved-source", "cycle"],
)
def test_supersession_rejects_invalid_predecessor_record(policies, field, value, error):
    _validate_supersession(policies)
    changed = _replace_metadata(policies, "REMOTE-001-FR-V1", **{field: value})
    with pytest.raises(AssertionError, match=error):
        _validate_supersession(changed)


@pytest.mark.parametrize(
    ("path", "overrides", "error"),
    [
        ("en/DATA-001-EN-V1.md", {"language": "en", "document_id": "DATA-001-EN-V1"}, "Unauthorized policy slot"),
        ("en/REMOTE-001-FR-V1.md", {"language": "en"}, "Duplicate document ID"),
        ("fr/OTHER-REMOTE-FR-V1.md", {"document_id": "OTHER-REMOTE-FR-V1"}, "Duplicate policy slot"),
    ],
    ids=["seventh-historical-slot", "duplicate-id", "duplicate-slot"],
)
def test_inventory_rejects_invalid_historical_additions(corpus, path, overrides, error):
    assert _validate_inventory(corpus) == set()
    family = "DATA" if "DATA-" in path else "REMOTE"
    fields, body = corpus[Path(f"fr/{family}-001-FR-V1.md")]
    changed = corpus | {Path(path): (fields | overrides, body)}
    with pytest.raises(AssertionError, match=error):
        _validate_inventory(changed)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("allowed_roles", "[hr, employee]", "Unexpected allowed_roles"),
        ("classification", "INTERNAL", "Unexpected classification"),
        ("approval_status", "superseded", "Unexpected approval_status"),
        ("expires_at", '"null"', "Unexpected expires_at"),
        ("supersedes", '"null"', "Unexpected supersedes"),
        ("effective_date", '"2026-09-01"', "Unexpected effective_date"),
        ("is_historical", "true", "Unexpected metadata fields"),
    ],
    ids=["acl", "classification", "invented-status", "quoted-expiry", "quoted-predecessor", "date", "extra-key"],
)
def test_historical_metadata_rejects_invalid_values(policies, field, value, error):
    document_id = "LEAVE-001-FR-V1"
    fields, _ = policies[document_id]
    _check_historical_metadata(document_id, fields)
    with pytest.raises(AssertionError, match=error):
        _check_historical_metadata(document_id, fields | {field: value})


def test_historical_metadata_rejects_duplicate_or_missing_keys():
    path = _policy_path("LEAVE-001-FR-V1")
    source = path.read_text(encoding="utf-8")
    _check_historical_metadata(path.stem, _parse_policy(source, path)[0])
    key_line = "allowed_roles: [hr]\n"
    assert source.count(key_line) == 1

    with pytest.raises(AssertionError, match="Duplicate front-matter field 'allowed_roles'"):
        _parse_policy(source.replace(key_line, key_line * 2, 1), path)
    with pytest.raises(AssertionError, match="Missing required metadata.*allowed_roles"):
        _parse_policy(source.replace(key_line, "", 1), path)


@pytest.mark.parametrize("document_id", ACTIVE_HASHES)
def test_original_active_policy_text_is_preserved(document_id):
    text = _policy_path(document_id).read_text(encoding="utf-8")
    assert sha256(text.encode("utf-8")).hexdigest() == ACTIVE_HASHES[document_id]


def test_reviewed_evaluation_candidates_are_preserved():
    text = (REPOSITORY_ROOT / DEV_DATASET).read_text(encoding="utf-8")
    assert sha256(text.encode("utf-8")).hexdigest() == BASELINE_HASHES[DEV_DATASET]


HISTORICAL_CLAUSES = {
    "REMOTE-001-FR-V1": [
        (2, "Le salarié attend l’accord écrit avant de télétravailler."),
        (4, "Les règles de gestion des données d’Atlas Services s’appliquent aussi à distance."),
        (4, "partage les données uniquement avec les destinataires autorisés"),
        (5, "le responsable hiérarchique peut autoriser un jour supplémentaire après consultation des RH et avant le début de l’arrangement."),
    ],
    "REMOTE-001-EN-V1": [
        (2, "The employee waits for written approval before working remotely."),
        (4, "Atlas Services' data handling rules also apply to remote work."),
        (4, "share data only with authorized recipients"),
        (5, "the line manager may authorize an additional day after consulting HR and before the arrangement begins."),
    ],
    "LEAVE-001-FR-V1": [
        (2, "Il remet directement aux RH les justificatifs confidentiels."),
        (3, "Si le solde couvre la période demandée et que le responsable hiérarchique confirme la continuité du service, les RH peuvent approuver le congé et en informer le salarié."),
        (4, "Les RH seules peuvent consulter le registre de traitement des congés, les soldes individuels et les justificatifs."),
        (4, "Dates et statut nécessaires à la couverture du service"),
        (5, "les RH peuvent autoriser l’absence sans respecter le préavis ordinaire ni attendre la confirmation préalable du responsable hiérarchique."),
        (5, "Le salarié complète son dossier auprès des RH à son retour."),
    ],
    "EXPENSES-001-FR-V1": [
        (1, "Les dépenses personnelles, celles que l’entreprise a déjà réglées et celles déjà remboursées sont exclues."),
        (3, "dans les dix jours ouvrés suivant la date de la dépense."),
        (3, "Pour les dépenses de déplacement professionnel, le délai fixé par la politique de déplacements professionnels s’applique."),
        (4, "Si le coût estimé des achats opérationnels dépasse 1 000 MAD, le responsable hiérarchique et Finance doivent approuver l’achat avant tout engagement de la dépense."),
        (4, "Finance contrôle les justificatifs, les plafonds et l’absence de remboursement antérieur."),
        (5, "Finance peut accepter une copie délivrée par le fournisseur, avec la preuve du paiement et une explication de l’absence de l’original."),
    ],
    "ACCESS-001-FR-V1": [
        (2, "Si le responsable hiérarchique a validé le besoin et que le périmètre est justifié, l’accès peut être attribué après approbation de la Sécurité informatique."),
        (3, "La revue V1 est déclenchée par le signalement d’un changement ou de la fin du besoin."),
        (4, "la Sécurité informatique retire l’accès sur notification du service responsable"),
        (5, "Son approbation est requise avant l’attribution."),
        (5, "La Sécurité informatique consigne le périmètre autorisé et retire les droits à l’issue de la tâche."),
    ],
    "DATA-001-FR-V1": [
        (2, "| RESTRICTED | Autorisation explicite pour le document | Utiliser l’espace approuvé et limiter le partage au strict nécessaire |"),
        (2, "Si l’autorisation du destinataire est inconnue, le salarié suspend le partage"),
        (3, "Si un transfert externe est nécessaire, le propriétaire des données peut l’autoriser par écrit après avoir vérifié le besoin et l’autorisation du destinataire."),
        (3, "Le salarié utilise uniquement un canal approuvé pour les données concernées."),
        (4, "informe rapidement la Sécurité informatique par le canal interne prévu"),
        (5, "le propriétaire des données peut autoriser par écrit le dépôt dans un espace déjà approuvé pour ces données, avec accès limité aux mêmes destinataires autorisés."),
    ],
}


@pytest.mark.parametrize("document_id", HISTORICAL_METADATA)
def test_historical_policy_has_sections_table_and_versioned_workflow(policies, document_id):
    _, body = policies[document_id]
    sections = _sections(body)
    numbers = re.findall(r"^## ([1-9]\d*)\. \S", body, re.MULTILINE)
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

    approval = "Conditional approval" if "-EN-" in document_id else "Approbation conditionnelle"
    assert f"**{approval}.** " in body and "**Exception.** " in body
    assert not [marker for marker in PROHIBITED_MARKERS if marker.casefold() in body.casefold()]
    for number, clause in HISTORICAL_CLAUSES[document_id]:
        assert clause in sections[number][1], f"{document_id} §{number}: {clause}"


@pytest.mark.parametrize(
    ("language", "pattern", "expected"),
    [
        ("FR", r"([a-z]+) jours? de télétravail par semaine", ["un", "deux"]),
        ("EN", r"([a-z]+) remote-work days? per week", ["one", "two"]),
    ],
    ids=["fr", "en"],
)
def test_remote_work_weekly_limit_is_one_in_v1_and_two_in_v2(policies, language, pattern, expected):
    for version, limit in enumerate(expected, 1):
        _, body = policies[f"REMOTE-001-{language}-V{version}"]
        authorization = _sections(body)[2][1]
        assert re.findall(pattern, authorization) == [limit]


def test_leave_notice_is_seven_working_days_in_v1_and_five_in_v2(policies):
    for version, notice in [(1, "sept"), (2, "cinq")]:
        _, body = policies[f"LEAVE-001-FR-V{version}"]
        section = _sections(body)[2][1]
        assert re.findall(r"au moins ([a-z]+) jours ouvrés avant le départ", section) == [notice]


def test_expenses_field_meal_cap_is_120_in_v1_and_150_in_v2(policies):
    pattern = (
        r"^\| Repas en mission sur le terrain \| Plafond de (\d+) MAD par repas "
        r"si l’entreprise ne fournit pas le repas dans le cadre de la mission \| Reçu détaillé \|$"
    )
    for version, cap in [(1, "120"), (2, "150")]:
        _, body = policies[f"EXPENSES-001-FR-V{version}"]
        assert re.findall(pattern, _sections(body)[2][1], re.MULTILINE) == [cap]


def test_registry_lists_all_policies_and_exact_historical_provenance(corpus, policies):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    _validate_identity_registry(matrix, corpus)
    identities = _registry_rows(matrix, "E", "F", IDENTITY_COLUMNS)
    assert {row[0].strip("\x60") for row in identities} == policies.keys()

    columns = ("Document ID", "Effective date", "Approval status", "Source type", "Source license")
    rows = _registry_rows(matrix, "H", "I", columns)
    provenance = {row[0].strip("\x60"): tuple(cell.strip("\x60") for cell in row[1:]) for row in rows}
    assert len(rows) == len(provenance) == 6
    assert provenance.keys() == HISTORICAL_METADATA.keys()
    for document_id, record in provenance.items():
        fields, _ = policies[document_id]
        assert record == (
            _nullable(fields["effective_date"]), fields["approval_status"],
            fields["source_type"], fields["source_license"],
        )


HISTORICAL_REFERENCES = {
    "Remote Work weekly limit": {
        ("REMOTE-001-FR-V1", 2),
        ("REMOTE-001-EN-V1", 2),
        ("REMOTE-001-FR-V2", 2),
        ("REMOTE-001-EN-V2", 2),
    },
    "Remote Work historical data duties and exception": {
        ("REMOTE-001-FR-V1", 4),
        ("REMOTE-001-FR-V1", 5),
        ("REMOTE-001-EN-V1", 4),
        ("REMOTE-001-EN-V1", 5),
        ("REMOTE-001-FR-V2", 5),
        ("REMOTE-001-EN-V2", 5),
    },
    "Leave FR planned notice": {
        ("LEAVE-001-FR-V1", 2),
        ("LEAVE-001-FR-V2", 2),
    },
    "Leave FR historical approval, confidentiality and emergency": {
        ("LEAVE-001-FR-V1", 3),
        ("LEAVE-001-FR-V1", 4),
        ("LEAVE-001-FR-V1", 5),
        ("LEAVE-001-FR-V2", 4),
        ("LEAVE-001-FR-V2", 5),
    },
    "Expenses FR field meal cap": {
        ("EXPENSES-001-FR-V1", 2),
        ("EXPENSES-001-FR-V2", 2),
    },
    "Expenses FR historical documentation and review": {
        ("EXPENSES-001-FR-V1", 3),
        ("EXPENSES-001-FR-V1", 4),
        ("EXPENSES-001-FR-V1", 5),
    },
    "Access Control FR review and temporary grants": {
        ("ACCESS-001-FR-V1", 2),
        ("ACCESS-001-FR-V1", 3),
        ("ACCESS-001-FR-V1", 4),
        ("ACCESS-001-FR-V1", 5),
        ("ACCESS-001-FR-V2", 3),
        ("ACCESS-001-FR-V2", 4),
        ("ACCESS-001-FR-V2", 5),
    },
    "Data Handling FR transfer and channel approval": {
        ("DATA-001-FR-V1", 2),
        ("DATA-001-FR-V1", 3),
        ("DATA-001-FR-V1", 4),
        ("DATA-001-FR-V1", 5),
        ("DATA-001-FR-V2", 3),
        ("DATA-001-FR-V2", 5),
    },
}


def test_historical_registry_rules_cite_v1_and_the_relevant_v2_sections(policies):
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    rows = _registry_rows(matrix, "I", "J", RULE_COLUMNS)
    registered = {rule: (requirement, source) for rule, requirement, source in rows}
    assert len(rows) == len(registered)
    assert HISTORICAL_REFERENCES.keys() <= registered.keys()
    for rule, references in HISTORICAL_REFERENCES.items():
        _, source = registered[rule]
        assert _validate_source_references(source, policies) == references, rule

    assert "one day per week" in registered["Remote Work weekly limit"][0]
    assert "two days per week" in registered["Remote Work weekly limit"][0]
    assert "seven working days" in registered["Leave FR planned notice"][0]
    assert "five working days" in registered["Leave FR planned notice"][0]
    assert "120 MAD per meal" in registered["Expenses FR field meal cap"][0]
    assert "150 MAD per meal" in registered["Expenses FR field meal cap"][0]


def test_registered_active_rules_and_citations_are_preserved():
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    rules = _registry_rows(matrix, "A", "B", RULE_COLUMNS)
    original_names = REQUIRED_RULE_REFERENCES.keys() | BATCH_TWO_REFERENCES.keys()
    original_rules = sorted(row for row in rules if row[0] in original_names)
    assert sha256(json.dumps(original_rules).encode("utf-8")).hexdigest() == CURRENT_RULES_HASH
