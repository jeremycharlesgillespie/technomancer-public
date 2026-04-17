"""Tracing — end-to-end correlation IDs across Discord → executor → Jira.

Provides ULID-based trace IDs stored in a :class:`contextvars.ContextVar`
so they flow through async boundaries and can be propagated across process
boundaries (e.g. attached to HTTP headers or subprocess env vars) to
correlate activity end-to-end.

ULID format (26-char Crockford base32):
    - first 10 chars  : 48-bit millisecond timestamp (lexicographically sortable)
    - trailing 16 chars: 80 bits of cryptographic randomness

Crockford base32 alphabet excludes ``I``, ``L``, ``O``, ``U`` to avoid
visual ambiguity. Generation uses only stdlib (``secrets`` + ``time``).
"""

from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional


DEFAULT_TRACE_ID = "-"

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_SET = frozenset(_CROCKFORD_ALPHABET)

_ULID_LENGTH = 26
_TIMESTAMP_CHARS = 10
_RANDOM_CHARS = 16
_TIMESTAMP_BITS = 48
_RANDOM_BITS = 80

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default=DEFAULT_TRACE_ID)


def _encode_crockford(value: int, length: int) -> str:
    """Encode ``value`` as Crockford base32, MSB-first, zero-padded to ``length``."""
    if value < 0:
        raise ValueError("cannot encode negative integer")
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_trace_id() -> str:
    """Generate a fresh 26-char Crockford base32 ULID."""
    ts_ms = int(time.time() * 1000) & ((1 << _TIMESTAMP_BITS) - 1)
    rand_bits = secrets.randbits(_RANDOM_BITS)
    return _encode_crockford(ts_ms, _TIMESTAMP_CHARS) + _encode_crockford(
        rand_bits, _RANDOM_CHARS
    )


def get_trace_id() -> str:
    """Return the trace_id bound to the current context (or the sentinel)."""
    return _trace_id_var.get()


def set_trace_id(value: Optional[str]) -> Token[str]:
    """Bind ``value`` as the trace_id for the current context.

    Returns the :class:`Token` so callers can reset via
    ``_trace_id_var.reset(token)``. A ``None`` value stores the sentinel
    :data:`DEFAULT_TRACE_ID`.
    """
    return _trace_id_var.set(value if value else DEFAULT_TRACE_ID)


@contextmanager
def with_trace_id(value: Optional[str] = None) -> Iterator[str]:
    """Bind a trace_id for the body of the ``with`` block; auto-reset on exit.

    If ``value`` is falsy, a fresh ULID is generated. Yields the bound
    trace_id so callers can read or forward it without a separate
    :func:`get_trace_id` call.
    """
    trace_id = value if value else new_trace_id()
    token = _trace_id_var.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id_var.reset(token)


def is_valid_trace_id(value: object) -> bool:
    """Return True iff ``value`` is a 26-char uppercase Crockford base32 string."""
    if not isinstance(value, str):
        return False
    if len(value) != _ULID_LENGTH:
        return False
    return all(c in _CROCKFORD_SET for c in value)
