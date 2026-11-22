"""Read the actual phase contracts without datasets, reports or model runtime."""

import ast
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODES = ("BM25S", "E5/FAISS", "Hybrid RRF", "Hybrid + CE")


def section(document, number):
    match = re.search(rf"^## {number}\. [^\n]*\n(.*?)(?=^## \d+\.|\Z)", document, re.M | re.S)
    assert match, f"Missing contract section {number}"
    return match[1]


def subsection(document, heading):
    match = re.search(rf"^### {re.escape(heading)}\n(.*?)(?=^### |\Z)", document, re.M | re.S)
    assert match, f"Missing gate {heading}"
    return match[1]


def normalized(text):
    return " ".join(text.replace("*", "").replace("`", "").split()).lower()


def require(text, *patterns):
    for pattern in patterns:
        assert re.search(pattern, normalized(text)), f"Missing documented requirement: {pattern}"


def require_modes(text):
    for mode in MODES:
        assert mode.lower() in normalized(text), f"Missing evaluation configuration: {mode}"


def development_gate(text):
    require(text, r"\b20(?:-case)? development\b", r"four-way", r"reproducib", r"retrieval-readiness",
            r"role-isolation/security", r"owner-approved reranker inclusion decision")
    require_modes(text)


def frozen_gate(text):
    require(text, r"\b60 frozen held-out cases\b", r"four-way", r"make eval-retrieval",
            r"tests/eval/test_ablation\.py", r"eval/report\.json", r"end-to-end generation",
            r"citation", r"refusal", r"adversarial security", r"multilingual", r"measured.*latency",
            r"provenance", r"reproducib")
    require_modes(text)


def read_contracts():
    return ((ROOT / "docs/decisions.md").read_text(encoding="utf-8"),
            (ROOT / "AGENTS.md").read_text(encoding="utf-8"))


def test_section_reader_preserves_content_and_stops_at_next_numbered_heading():
    document = "## 11. Matrix\nrequired gate\n### Detail\nretained\n## 12. Security\nseparate gate\n"
    assert section(document, 11) == "required gate\n### Detail\nretained\n"
    assert section(document, 12) == "separate gate\n"


@pytest.fixture(scope="module")
def contracts():
    return read_contracts()


def current_gates(contracts):
    decisions, agents = contracts
    matrix = section(decisions, 11)
    order = section(decisions, 15)
    agent_order = section(agents, 18)
    decision_lines = re.findall(r"^\*\*Phase ([1-8]) — .*", order, re.M)
    agent_lines = re.findall(r"^([1-8])\. \*\*.*", agent_order, re.M)
    assert decision_lines == agent_lines == list("12345678")
    line = lambda text, pattern: re.search(pattern, text, re.M)[0]
    return (
        subsection(matrix, "Phase 5 — Development retrieval gate"),
        # Inspect the binding gate prose, not incidental words in the general
        # feature catalog that follows it (e.g. a future citation test path).
        subsection(matrix, "Phase 7 — Frozen retrieval gate").split("\n| Feature |", 1)[0],
        line(order, r"^\*\*Phase 5 — .*"), line(order, r"^\*\*Phase 7 — .*"),
        line(agent_order, r"^5\. .*"), line(agent_order, r"^7\. .*"),
    )


def test_phase5_requires_all_twenty_approved_cases_and_four_configurations(contracts):
    dev, _, phase5, _, agent5, _ = current_gates(contracts)
    development_gate(dev)
    development_gate(agent5)
    require(phase5, r"full approved 20-case development four-way retrieval comparison")
    assert "approved" in normalized(dev)


def test_phase5_retains_readiness_security_and_owner_decision(contracts):
    dev, _, phase5, _, agent5, _ = current_gates(contracts)
    for text in (dev, phase5, agent5):
        require(text, r"retrieval-readiness", r"reproducib", r"role-isolation/security",
                r"documented.*owner-approved reranker inclusion decision")
    require(dev, r"zero unauthorized retrieval", r"measured latency", r"tests must pass normally")
    readiness = subsection(section(contracts[0], 17), "Retrieval-readiness milestone")
    require(readiness, r"at least 20 documents have been ingested and validated",
            r"snapshot hash is reproducible across clean runs",
            r"bm25 baseline exists with recall@5 strictly greater than 0")


def test_phase7_retains_sixty_frozen_cases_all_modes_and_executable_acceptance(contracts):
    _, frozen, _, phase7, _, agent7 = current_gates(contracts)
    frozen_gate(frozen)
    frozen_gate(agent7)
    for text in (frozen, phase7, agent7):
        require(text, r"60 frozen held-out cases", r"make eval-retrieval.*succe",
                r"tests/eval/test_ablation\.py.*pass", r"eval/report\.json.*generat|generat.*eval/report\.json")
    assert "| Retrieval ablation | Four configs run on frozen benchmark with machine-readable `eval/report.json`. | `tests/eval/test_ablation.py` |" in section(contracts[0], 11)


def test_identical_conditions_and_all_later_quality_security_obligations_remain(contracts):
    dev, frozen, _, phase7, _, agent7 = current_gates(contracts)
    for text in (dev, frozen):
        require(text, r"identical snapshot, labels, roles and query conditions")
    for text in (frozen, phase7, agent7):
        require(text, r"end-to-end generation", r"citation", r"refusal", r"adversarial security",
                r"multilingual", r"latency", r"provenance", r"reproducib")
    require(frozen, r"semantic citation support", r"refusal accuracy", r"p50/p95", r"declared hardware",
            r"security thresholds.*release requirements remain mandatory")
    for text in (section(contracts[0], 10), section(contracts[1], 14)):
        require(text, r"recall@5", r"mrr@10", r"ndcg@10", r"refusal accuracy", r"unauthorized retrieval", r"p50/p95")


def test_full_scope_counts_and_pairs_are_unchanged_in_both_contracts(contracts):
    decisions, agents = contracts
    full = subsection(section(decisions, 10), "Full scope — 80 query-role cases")
    counts = {}
    for line in full.splitlines():
        cells = [cell.strip().replace("*", "") for cell in line.split("|")[1:-1]]
        if len(cells) == 2 and cells[0] in ("Answerable", "Unanswerable", "Role-sensitive executions", "Total"):
            counts[cells[0]] = int(re.match(r"\d+", cells[1])[0])
    assert counts == {"Answerable":48, "Unanswerable":16, "Role-sensitive executions":16, "Total":80}
    assert sum(counts[k] for k in ("Answerable", "Unanswerable", "Role-sensitive executions")) == counts["Total"]
    require(full, r"8 authorized/unauthorized pairs", r"20 development \+ 60 frozen held-out")
    full_agent = section(agents, 14).split("**Reduced scope:**", 1)[0]
    require(full_agent, r"80 query-role executions", r"48 answerable", r"16 unanswerable",
            r"16 role-sensitive executions.*eight authorized/unauthorized pairs",
            r"20 development cases, 60 frozen held-out cases")


def test_step117_is_only_a_sequencing_approval_and_activates_no_fallback(contracts):
    approval = subsection(section(contracts[0], 17), "Step 117 owner approval — evaluation-gate sequencing only")
    require(approval, r"project owner explicitly approved", r"limited to phase sequencing",
            r"separate from step 59", r"step 90/103", r"step 116",
            r"80 query-role executions: 20 development \+ 60 frozen held-out",
            r"48 answerable \+ 16 unanswerable \+ 16 role-sensitive executions",
            r"no reduced-scope fallback is activated", r"no evaluation scope reduction",
            r"label change", r"architecture/configuration change", r"skipped/xfail mandatory test")
    require(section(contracts[1], 18), r"changes only evaluation-gate sequencing",
            r"full scope remains 80 executions", r"no reduced-scope fallback is activated")


def test_step116_architecture_decision_is_preserved_as_approved_history(contracts):
    decisions, _ = contracts
    owner = subsection(section(decisions, 2), "Step 116 owner decision — v1 retrieval baseline")
    block, = re.findall(r"```json\n(.*?)\n```", owner, re.S)
    record = json.loads(block)
    assert record["owner_approved"] is True and record["deployment_baseline"] == "hybrid"
    assert record["production_reranker_enabled"] is False
    assert record["retrieval_configuration_change_approved"] is False
    assert record["evaluation_modes"] == ["bm25", "faiss", "hybrid", "hybrid_rerank"]
    assert record["faiss_only_selection"] == record["embedding_fine_tuning"] == "DEFERRED_UNTIL_AFTER_V1_PORTFOLIO_RELEASE"
    assert record["phase5_status"] == "OPEN" and record["contractual_frozen_ablation_complete"] is False
    require(owner, r"historical audit", r"step 117.*supersedes the interpretation",
            r"hybrid/ce architecture decision remains unchanged", r"does not declare the current phase 5 closed")


def test_development_evaluator_cannot_claim_frozen_completion(contracts):
    # Inspect actual evaluator constants without importing torch, FAISS, any
    # checkpoint loader, the evaluator runner, or an ignored benchmark report.
    tree = ast.parse((ROOT / "eval/dev_four_way_diagnostic.py").read_text(encoding="utf-8"))
    for field, expected in (("full_ablation_complete", False), ("held_out_evaluated_count", 0)):
        values = [node.value for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == field]
        assert len(values) == 1 and isinstance(values[0], ast.Constant)
        assert type(values[0].value) is type(expected) and values[0].value == expected
    require(section(contracts[0], 11), r"completed development diagnostic is not a completed frozen evaluation",
            r"do not relabel development-only evidence", r"report pending phase 7 requirements as passed")
    require(section(contracts[1], 18), r"completed development diagnostic is not a completed frozen evaluation")


def test_phase_advance_requires_passing_gates_and_explicit_human_review(contracts):
    current_gates(contracts)  # Enforce all eight ordered phases in each file.
    require(section(contracts[1], 18), r"current-phase acceptance gates must pass before advancing",
            r"does not automatically close phase 5 or authorize a phase transition")
    approval = subsection(section(contracts[0], 17), "Step 117 owner approval — evaluation-gate sequencing only")
    require(approval, r"phase advances only after its applicable acceptance gates pass",
            r"neither automatic phase 5 closure nor a phase transition or phase 6 implementation",
            r"phase 5 ready for human closure review.*only if every applicable phase 5 gate passes",
            r"phase 5 open", r"phase 7 remains pending until its own gates execute and pass")


def test_contract_reading_has_no_dataset_report_or_model_io(monkeypatch):
    original = Path.read_text
    reads = []
    allowed = {ROOT/"AGENTS.md", ROOT/"docs/decisions.md"}
    def observe(path, *args, **kwargs):
        assert path in allowed, "Contract verification may only read the two project contracts"
        reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", observe)
    current_gates(read_contracts())
    assert len(reads) == 2 and set(reads) == allowed


@pytest.mark.parametrize("defect", ["dev_count", "owner", "readiness", "security", "frozen_count",
    "mode", "make", "test", "report", "citation", "refusal", "provenance"])
def test_gate_checks_reject_reduced_or_missing_requirements(contracts, defect):
    dev, frozen, *_ = current_gates(contracts)
    replacements = {
        "dev_count": ("20 development", "19 development"),
        "owner": ("owner-approved reranker inclusion decision", "optional decision"),
        "readiness": ("retrieval-readiness", "optional readiness"),
        "security": ("role-isolation/security", "optional checks"),
        "frozen_count": ("60 frozen held-out cases", "59 frozen held-out cases"),
        "mode": ("Hybrid + CE", "three-mode comparison"),
        "make": ("make eval-retrieval", "optional runner"),
        "test": ("tests/eval/test_ablation.py", "optional check"),
        "report": ("eval/report.json", "optional report"),
        "citation": ("citation", "optional grounding"),
        "refusal": ("refusal", "optional response"),
        "provenance": ("provenance", "optional metadata"),
    }
    before, after = replacements[defect]
    if defect in ("dev_count", "owner", "readiness", "security"):
        assert before in dev
        with pytest.raises(AssertionError):
            development_gate(dev.replace(before, after))
    else:
        assert before in frozen
        with pytest.raises(AssertionError):
            frozen_gate(frozen.replace(before, after))
