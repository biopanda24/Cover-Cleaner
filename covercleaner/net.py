"""HTTPS for update checks and model downloads.

Python's own certificate check on Windows only sees the root certificates
Windows has already downloaded, so some sites (GitHub rotates between
certificate chains) fail at random with CERTIFICATE_VERIFY_FAILED.
truststore checks certificates the way the operating system and browsers
do, so it is used when it is installed (requirements.txt has it).
"""
from __future__ import annotations

import ssl
import urllib.request

from . import __version__

USER_AGENT = f"CoverCleaner/{__version__}"
_ctx = None


def ssl_context() -> ssl.SSLContext:
    global _ctx
    if _ctx is None:
        try:
            import truststore
            _ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except Exception:                       # not installed or unsupported here
            _ctx = ssl.create_default_context()
    return _ctx


def urlopen(url: str, timeout: float, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    if url.startswith("https:"):
        return urllib.request.urlopen(req, timeout=timeout, context=ssl_context())
    return urllib.request.urlopen(req, timeout=timeout)
