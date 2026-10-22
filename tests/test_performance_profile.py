import json

from backend.main import app
from modules.pipeline import _fingerprint, _system_snapshot, _write_json_atomic


def test_media_fingerprint_changes_when_source_bytes_change(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    first = _fingerprint(source)
    source.write_bytes(b"second")
    assert _fingerprint(source) != first


def test_atomic_json_writer_creates_parent_and_complete_file(tmp_path):
    destination = tmp_path / "nested" / "cache.json"
    _write_json_atomic(destination, {"state": "complete", "values": [1, 2, 3]})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "state": "complete", "values": [1, 2, 3]
    }
    assert list(destination.parent.iterdir()) == [destination]


def test_resource_profile_has_stable_fields():
    profile = _system_snapshot()
    assert {"cpu_percent", "ram_used_bytes", "process_rss_bytes",
            "gpu_available", "gpu_memory_used_bytes",
            "gpu_memory_allocated_bytes"}.issubset(profile)


def test_existing_fastapi_routes_remain_registered():
    paths = {route.path for route in app.routes}
    assert {"/health", "/upload", "/summarize/{job_id}", "/status/{job_id}",
            "/result/{job_id}", "/coverage/{job_id}"}.issubset(paths)
