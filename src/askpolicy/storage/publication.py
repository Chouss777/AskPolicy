"""Single-writer, same-filesystem publication with pinned offline reader handles."""

from contextlib import contextmanager
import os
from pathlib import Path
from uuid import uuid4

from .index_store import (
    IndexStoreError, PublishedSnapshot, canonical, digest, load_snapshot,
    read_json, safe_path, stage_snapshot, valid_id,
)


@contextmanager
def _writer(root):
    path = root / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise IndexStoreError("PUBLICATION_BUSY") from None
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink()


def _switch_pointer(root: Path, handle: PublishedSnapshot) -> None:
    temporary = root / (".active-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(dict(format=1, release_id=handle.release_id, manifest_hash=handle.manifest_hash)))
            stream.flush()
            os.fsync(stream.fileno())
        # Both files share the same parent and filesystem. Windows supports
        # replacing this closed file, without replacing any populated directory.
        os.replace(temporary, root / "active.json")
    finally:
        temporary.unlink(missing_ok=True)


def publish_snapshot(connection, root: Path, *, manifest, snapshot_config, validity_date, encoder) -> PublishedSnapshot:
    """Stage, reload, finalize, then atomically replace the sole active pointer.

    Only this writer creates releases. Finalized release paths are never written
    again or removed. A crashed writer's exclusive lock requires operator recovery
    after confirming that writer has stopped. Readers never recover it implicitly.
    """
    try:
        root = safe_path(root)
        root.mkdir(parents=True, exist_ok=True)
        options = dict(manifest=manifest, snapshot_config=snapshot_config, validity_date=validity_date, encoder=encoder)
        with _writer(root):
            staging = root / "staging" / uuid4().hex
            release_id = stage_snapshot(connection, staging, **options)
            handle = load_snapshot(staging, expected_release_id=release_id, **options)
            releases = safe_path(root / "releases")
            releases.mkdir(exist_ok=True)
            final = safe_path(releases / release_id)
            if final.exists():
                handle = load_snapshot(final, expected_release_id=release_id, **options)
            else:
                # Rename only into a nonexistent path under the writer lock.
                staging.rename(final)
            _switch_pointer(root, handle)
            return handle
    except IndexStoreError:
        raise
    except Exception:
        raise IndexStoreError("PUBLICATION_FAILED") from None


def load_active_snapshot(root: Path, *, manifest, snapshot_config, validity_date, encoder) -> PublishedSnapshot:
    """Read the pointer once and load exactly that release, with no fallback."""
    try:
        root = safe_path(root)
        pointer = read_json(safe_path(root / "active.json").read_bytes())
        if (
            type(pointer) is not dict or set(pointer) != {"format", "release_id", "manifest_hash"}
            or type(pointer["format"]) is not int or pointer["format"] != 1
            or not valid_id(pointer["release_id"]) or not valid_id(pointer["manifest_hash"])
        ):
            raise IndexStoreError("INVALID_ACTIVE_POINTER")
        directory = safe_path(root / "releases" / pointer["release_id"])
        if digest(safe_path(directory / "manifest.json").read_bytes()) != pointer["manifest_hash"]:
            raise IndexStoreError("MANIFEST_HASH_MISMATCH")
        handle = load_snapshot(directory, manifest=manifest, snapshot_config=snapshot_config,
                               validity_date=validity_date, encoder=encoder, expected_release_id=pointer["release_id"])
        if handle.manifest_hash != pointer["manifest_hash"]:
            raise IndexStoreError("MANIFEST_HASH_MISMATCH")
        return handle
    except IndexStoreError:
        raise
    except Exception:
        raise IndexStoreError("ACTIVE_LOAD_FAILED") from None
