"""Lightweight trusted configuration and UTC clock contracts."""

from datetime import UTC,datetime,date
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from security.test_auth import secured


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"src"))
    from askpolicy.api import retrieval_runtime
    return retrieval_runtime


def test_environment_configuration_and_utc_default(runtime,tmp_path):
    env={"ASKPOLICY_RELEASE_ROOT":str(tmp_path/"releases"),"ASKPOLICY_E5_DIR":str(tmp_path/"model"),
         "ASKPOLICY_MANIFEST_PATH":str(tmp_path/"manifest.json")}
    config=runtime.load_retrieval_config(env)
    assert config.release_root==tmp_path/"releases"
    assert runtime.load_retrieval_config({}) is None
    assert runtime.utc_today()==datetime.now(UTC).date()
    assert type(runtime.utc_today()) is date


@pytest.mark.parametrize("value", ["relative/path","","  /path","../escape",None,1])
def test_malformed_paths_rejected(runtime,tmp_path,value):
    with pytest.raises(ValueError):
        runtime.RetrievalConfig(release_root=value,e5_dir=tmp_path/"model",manifest_path=tmp_path/"manifest")


def test_zero_argument_factory_uses_process_config_and_no_dotenv(runtime,tmp_path,monkeypatch,secured):
    from askpolicy.api.app import create_app
    from integration.test_retrieval_api import request_api
    seen=[]
    for name,value in (("JWT_SECRET",secured.key),("JWT_ISSUER",secured.config.issuer),
                       ("JWT_AUDIENCE",secured.config.audience),
                       ("ASKPOLICY_RELEASE_ROOT",str(tmp_path/"releases")),
                       ("ASKPOLICY_E5_DIR",str(tmp_path/"model")),
                       ("ASKPOLICY_MANIFEST_PATH",str(tmp_path/"manifest"))):
        monkeypatch.setenv(name,value)
    def observe(self,context,query):
        seen.append(self.config)
        raise runtime.RetrievalUnavailable("RETRIEVAL_UNAVAILABLE")
    monkeypatch.setattr(runtime.RetrievalRuntime,"search",observe)
    result=request_api(create_app(),secured.sign(),"employee")
    assert result.status==503 and seen[0].release_root==tmp_path/"releases"
