"""Pytest configuration ensuring deterministic offline regression tests do not burn live cloud API quota."""
import os

# Keep regression test suite fast, deterministic, and isolated from live cloud API keys in `.env`
os.environ.setdefault("SPATIAL_LLM_MODE", "from_scratch")

