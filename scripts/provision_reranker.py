"""Explicit five-file HTTPS provisioning outside Git, with atomic finalization."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import ssl
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from askpolicy.retrieval.cross_encoder import (
    ARTIFACTS, MODEL_ID, MODEL_REVISION, CrossEncoderError, safe_path, verify_checkpoint,
)


class SecureRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise CrossEncoderError("INSECURE_DOWNLOAD_REDIRECT")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def provision(destination: Path) -> dict:
    destination = safe_path(destination)
    if destination.name != MODEL_REVISION:
        raise CrossEncoderError("MODEL_REVISION_MISMATCH")
    if destination.resolve().is_relative_to(ROOT.resolve()) or any((p / ".git").exists() for p in destination.parents):
        raise CrossEncoderError("MODEL_PATH_INSIDE_GIT")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ("." + MODEL_REVISION + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise CrossEncoderError("PROVISIONING_BUSY") from None
    try:
        if destination.exists():
            return verify_checkpoint(destination)
        # The generated temporary root is verified inside the named cache parent
        # before its automatic recursive cleanup can run.
        with TemporaryDirectory(prefix=".reranker-", dir=destination.parent) as temporary:
            staging_root = safe_path(Path(temporary))
            if staging_root.parent.resolve() != destination.parent.resolve():
                raise CrossEncoderError("UNSAFE_STAGING_PATH")
            stage = staging_root / MODEL_REVISION
            stage.mkdir()
            opener = build_opener(HTTPSHandler(context=ssl.create_default_context()), SecureRedirect())
            for name, (expected_size, expected_hash) in ARTIFACTS.items():
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
                request = Request(url, headers={"Accept-Encoding":"identity"})
                checksum = sha256()
                size = 0
                with opener.open(request, timeout=60) as response, (stage / name).open("xb") as output:
                    if urlsplit(response.geturl()).scheme != "https":
                        raise CrossEncoderError("INSECURE_DOWNLOAD_TRANSPORT")
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > expected_size:
                            raise CrossEncoderError("DOWNLOAD_SIZE_MISMATCH")
                        checksum.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if size != expected_size or checksum.hexdigest() != expected_hash:
                    raise CrossEncoderError("DOWNLOAD_INTEGRITY_MISMATCH")
                print(json.dumps(dict(file=name, verified_bytes=size)), flush=True)
            records = verify_checkpoint(stage)
            if destination.exists():
                raise CrossEncoderError("DESTINATION_CHANGED")
            stage.rename(destination)
            return records
    except CrossEncoderError:
        raise
    except Exception:
        raise CrossEncoderError("PROVISIONING_FAILED") from None
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = provision(args.destination)
        print(json.dumps(dict(model_id=MODEL_ID, revision=MODEL_REVISION,
                              path=str(args.destination), artifacts=records), sort_keys=True))
    except CrossEncoderError as error:
        print(error.code, file=sys.stderr)
        raise SystemExit(1)
