import hashlib
import json


def stable_hash(obj: dict[str, object], length: int = 32) -> str:
    """Hash a fingerprint-dict."""
    assert length <= 64, "maximum 64 hex chars (256bit)"
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()[:length]
