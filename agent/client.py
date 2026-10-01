"""Groq wrapper: retry, SHA-256 file cache, JSON mode, and trace metadata."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Groq model aliases used by the existing application.
HAIKU = "openai/gpt-oss-20b"
SONNET = "openai/gpt-oss-120b"

TIMEOUT_S = 45.0

_client = None
_last_trace: dict = {}


def _get_client():
    global _client

    if _client is None:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        _client = Groq(
            api_key=api_key,
            timeout=TIMEOUT_S,
            max_retries=0,
        )

    return _client


def _key(model: str, system: str, user) -> str:
    if isinstance(user, str):
        user_text = user
    else:
        user_text = json.dumps(
            user,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    return hashlib.sha256(
        (model + "\x00" + system + "\x00" + user_text).encode()
    ).hexdigest()


def _strip_fences(text: str) -> str:
    text = text.strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)

    if m:
        text = m.group(1).strip()

    starts = [
        i for i in (text.find("{"), text.find("["))
        if i >= 0
    ]

    start = min(starts, default=0)

    return text[start:]


def _parse_json(text: str):
    return json.loads(_strip_fences(text))


def _call_api(
    model: str,
    system: str,
    user,
    temperature: float,
    max_tokens: int,
) -> str:

    if isinstance(user, str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        # Groq/OpenAI-style multimodal content.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = _get_client().chat.completions.create(**kwargs)

    return response.choices[0].message.content or ""


def _text_of(user) -> str:
    if isinstance(user, str):
        return user

    return "\n".join(
        b.get("text", "")
        for b in user
        if isinstance(b, dict)
        and b.get("type") == "text"
    )


def last_trace() -> dict:
    return dict(_last_trace)


def ask(
    model: str,
    system: str,
    user,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    use_cache: bool = True,
    cache_key: str | None = None,
):
    """Call Groq with retry + file cache.

    Returns text, or parsed JSON when json_mode=True.
    """

    global _last_trace

    if model == "haiku":
        model = HAIKU
    elif model == "sonnet":
        model = SONNET

    key = _key(
        model,
        system,
        user if cache_key is None else cache_key,
    )

    path = CACHE_DIR / f"{key}.json"
    t0 = time.time()

    if use_cache and path.exists():
        cached = json.loads(
            path.read_text(encoding="utf-8")
        )

        _last_trace = {
            "model": model,
            "ms": int((time.time() - t0) * 1000),
            "cached": True,
        }

        return (
            cached["parsed"]
            if json_mode and "parsed" in cached
            else cached["text"]
        )

    text = None
    err = None

    for attempt in range(2):
        try:
            text = _call_api(
                model,
                system,
                user,
                temperature,
                max_tokens,
            )
            break

        except Exception as e:
            err = e

            if attempt == 0:
                time.sleep(1)

    if text is None:

        if path.exists():
            cached = json.loads(
                path.read_text(encoding="utf-8")
            )

            _last_trace = {
                "model": model,
                "ms": int((time.time() - t0) * 1000),
                "cached": True,
            }

            return (
                cached["parsed"]
                if json_mode and "parsed" in cached
                else cached["text"]
            )

        raise RuntimeError(
            f"Groq call failed twice and no cache: {err}"
        )

    parsed = None

    if json_mode:
        try:
            parsed = _parse_json(text)

        except (json.JSONDecodeError, ValueError) as e:

            repair = (
                _text_of(user)
                + "\n\n[SYSTEM] Your previous reply was not valid JSON ("
                + str(e)[:120]
                + "). Return the same content as ONLY valid JSON: "
                  "escape quotes and newlines inside strings, "
                  "no prose, no code fences.\n\nPrevious reply:\n"
                + text[:6000]
            )

            text = _call_api(
                model,
                system + "\n\nReturn ONLY valid JSON.",
                repair,
                temperature,
                max_tokens,
            )

            parsed = _parse_json(text)

    record = {
        "model": model,
        "text": text,
        "ts": time.time(),
    }

    if parsed is not None:
        record["parsed"] = parsed

    path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    _last_trace = {
        "model": model,
        "ms": int((time.time() - t0) * 1000),
        "cached": False,
    }

    return parsed if json_mode else text


def ask_image(
    model: str,
    system: str,
    prompt: str,
    image_bytes: bytes,
    mime: str = "image/png",
    json_mode: bool = False,
    max_tokens: int = 2048,
):
    """ask() for one image."""

    image_data = base64.standard_b64encode(
        image_bytes
    ).decode()

    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{image_data}"
            },
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    return ask(
        model,
        system,
        content,
        json_mode=json_mode,
        max_tokens=max_tokens,
        cache_key=(
            prompt
            + "\x00"
            + hashlib.sha256(image_bytes).hexdigest()
        ),
    )


def selftest() -> None:
    t0 = time.time()

    reply = ask(
        HAIKU,
        "You are a terse assistant.",
        "Say 'Nyaya ready' in Hindi and English.",
        use_cache=False,
        max_tokens=64,
    )

    print(
        f"[{int((time.time() - t0) * 1000)} ms] {reply}"
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        if not os.environ.get("GROQ_API_KEY"):
            print(
                "GROQ_API_KEY not set",
                file=sys.stderr,
            )
            sys.exit(1)

        selftest()

    else:
        print(__doc__)



