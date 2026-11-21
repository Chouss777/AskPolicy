"""Synthetic artifact tests. Real checkpoint acceptance runs separately."""

from hashlib import sha256
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from askpolicy.retrieval import cross_encoder as module


class FixtureTokenizer:
    is_fast = True
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def __call__(self, queries, passages, **options):
        self.calls.append((tuple(queries), tuple(passages), options))
        rows = []
        for query, passage in zip(queries, passages, strict=True):
            count = len(query.split()) + len(passage.split()) + 4
            rows.append([0, len(query), *([1] * (count - 3)), 2])
        return {"input_ids":rows, "attention_mask":[[1] * len(row) for row in rows]}

    def pad(self, encoded, *, padding, return_tensors, verbose):
        assert padding and return_tensors == "pt" and not verbose
        size = max(map(len, encoded["input_ids"]))
        return {key:torch.tensor([row + [0] * (size-len(row)) for row in rows]) for key,rows in encoded.items()}


class FixtureModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type="xlm-roberta", architectures=["XLMRobertaForSequenceClassification"],
            hidden_size=384, num_hidden_layers=12, num_labels=1, max_position_embeddings=514)
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.options = None
        self.training = True
        self.calls = []
        self.output = None

    def to(self, **options):
        self.options = options
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return iter((self.parameter,))

    def __call__(self, **batch):
        assert not self.training and torch.is_inference_mode_enabled()
        assert all(value.device.type == "cpu" for value in batch.values())
        self.calls.append(len(batch["input_ids"]))
        logits = batch["input_ids"][:,1:2].float() if self.output is None else self.output
        return SimpleNamespace(logits=logits)


@pytest.fixture
def local(tmp_path, monkeypatch):
    path = tmp_path / module.MODEL_REVISION
    path.mkdir()
    data = {name:name.encode() for name in module.ARTIFACTS}
    artifacts = {name:(len(raw),sha256(raw).hexdigest()) for name,raw in data.items()}
    for name,raw in data.items():
        (path / name).write_bytes(raw)
    monkeypatch.setattr(module,"ARTIFACTS",artifacts)
    tokenizer,model = FixtureTokenizer(),FixtureModel()
    loads = []
    def load_tokenizer(directory, **kwargs):
        loads.append(("tokenizer",directory,kwargs))
        return tokenizer
    def load_model(directory, **kwargs):
        loads.append(("model",directory,kwargs))
        return model
    monkeypatch.setattr(module.AutoTokenizer,"from_pretrained",load_tokenizer)
    monkeypatch.setattr(module.AutoModelForSequenceClassification,"from_pretrained",load_model)
    return SimpleNamespace(path=path,data=data,artifacts=artifacts,tokenizer=tokenizer,model=model,loads=loads)


def test_exact_offline_loader_flags_and_cpu_configuration(local):
    encoder = module.CrossEncoder(local.path)
    assert encoder.revision == "1427fd652930e4ba29e8149678df786c240d8825"
    assert local.loads == [
        ("tokenizer",local.path,dict(use_fast=True,local_files_only=True,trust_remote_code=False)),
        ("model",local.path,dict(local_files_only=True,trust_remote_code=False,use_safetensors=True)),
    ]
    assert local.model.options == dict(device="cpu",dtype=torch.float32) and not local.model.training


@pytest.mark.parametrize("defect,code",[
    ("revision","MODEL_REVISION_MISMATCH"),("directory","MODEL_REVISION_MISMATCH"),
    ("missing","MODEL_FILES_MISSING"),("weights","MODEL_ARTIFACT_HASH_MISMATCH"),
    ("tokenizer","MODEL_ARTIFACT_HASH_MISMATCH"),("truncated","MODEL_ARTIFACT_SIZE_MISMATCH"),
    ("pickle","UNEXPECTED_MODEL_ARTIFACT"),
])
def test_artifacts_rejected_before_any_loading(local,defect,code):
    options = {}
    if defect == "revision":
        options["revision"] = "main"
    elif defect == "directory":
        local.path = local.path.rename(local.path.with_name("main"))
    elif defect == "missing":
        (local.path / "model.safetensors").unlink()
    elif defect == "pickle":
        (local.path / "pytorch_model.bin").write_bytes(b"unsafe")
    else:
        name = "tokenizer.json" if defect == "tokenizer" else "model.safetensors"
        original = (local.path / name).read_bytes()
        changed = original[:-1] if defect == "truncated" else b"x" * len(original)
        (local.path / name).write_bytes(changed)
    result = None
    with pytest.raises(module.CrossEncoderError,match=f"^{code}$"):
        result = module.CrossEncoder(local.path,**options)
    assert result is None and not local.loads and not local.model.calls


@pytest.mark.parametrize("field,value",[
    ("model_type","bert"),("architectures",["AutoModel"]),("hidden_size",768),
    ("num_hidden_layers",6),("num_labels",2),("max_position_embeddings",512),
])
def test_unexpected_config_cannot_infer(local,field,value):
    setattr(local.model.config,field,value)
    with pytest.raises(module.CrossEncoderError,match="^UNEXPECTED_MODEL_CONFIGURATION$"):
        module.CrossEncoder(local.path)
    assert not local.model.calls


def test_provisioning_whitelist_verified_and_idempotent(local,tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT / "scripts"))
    import provision_reranker as provisioner
    monkeypatch.setattr(provisioner,"ARTIFACTS",local.artifacts)
    calls = []
    class Reply(io.BytesIO):
        def geturl(self):
            return "https://huggingface.co/approved"
    def download(request, **kwargs):
        name = request.full_url.rsplit("/",1)[-1]
        calls.append(request.full_url)
        return Reply(local.data[name])
    monkeypatch.setattr(provisioner,"build_opener",lambda *args:SimpleNamespace(open=download))
    destination = tmp_path / "external" / module.MODEL_REVISION
    records = provisioner.provision(destination)
    assert len(calls) == 5 and all(module.MODEL_REVISION in url for url in calls)
    assert set(p.name for p in destination.iterdir()) == set(local.artifacts)
    assert records == module.verify_checkpoint(destination)
    assert provisioner.provision(destination) == records and len(calls) == 5
    (destination / "config.json").write_bytes(b"x" * len(local.data["config.json"]))
    with pytest.raises(module.CrossEncoderError,match="^MODEL_ARTIFACT_HASH_MISMATCH$"):
        provisioner.provision(destination)
    assert len(calls) == 5
    with pytest.raises(module.CrossEncoderError,match="^INSECURE_DOWNLOAD_REDIRECT$"):
        provisioner.SecureRedirect().redirect_request(None,None,302,"redirect",{},"http://unsafe")


def test_provisioning_mismatch_never_finalizes_or_replaces_existing(local,tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT / "scripts"))
    import provision_reranker as provisioner
    monkeypatch.setattr(provisioner,"ARTIFACTS",local.artifacts)
    class Reply(io.BytesIO):
        def geturl(self):
            return "https://huggingface.co/approved"
    monkeypatch.setattr(provisioner,"build_opener",lambda *a:SimpleNamespace(open=lambda *a,**kw:Reply(b"bad")))
    destination = tmp_path / "external" / module.MODEL_REVISION
    with pytest.raises(module.CrossEncoderError):
        provisioner.provision(destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.lock"))


def test_contract_pins_and_hash_allowlist_match():
    assert len(module.ARTIFACTS) == 5
    for path in (ROOT / "AGENTS.md", ROOT / "docs/decisions.md"):
        text = path.read_text(encoding="utf-8")
        assert module.MODEL_ID in text and module.MODEL_REVISION in text
        for name,(size,checksum) in module.ARTIFACTS.items():
            assert name in text and str(size) in text and checksum in text
