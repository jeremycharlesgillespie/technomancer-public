"""
Image Identification - Cascading approach for identifying images.

1. Local vision model (llava-llama3)
2. Claude API (fallback for complex cases)
"""

import time as _time
from datetime import datetime
from typing import Literal, TypedDict

import ollama

from .perf_monitor import record_llm_call as _record_perf

# Create explicit ollama client (needed on Windows)
_vision_client = ollama.Client(host="http://127.0.0.1:11434")

# Vision model for local analysis
VISION_MODEL = "llava-llama3"


class ImageIdentificationResult(TypedDict):
    method: Literal["vision_model", "claude"]
    result: str
    confidence: float


def log(msg: str) -> None:
    """Simple logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [ImageID] {msg}", flush=True)


def analyze_with_vision_model(image_bytes_list: list[bytes], prompt: str | None = None) -> str:
    """
    Use local vision model (llava-llama3) to analyze image.
    """
    if not prompt:
        prompt = """Analyze this image in detail. Include:
1. What/who is shown (identify characters, people, objects)
2. For anime/cartoon characters: name the character and series if recognizable
3. Notable visual details and style

Be specific - if you recognize a character, name them confidently."""

    start = _time.perf_counter()
    try:
        response = _vision_client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": image_bytes_list}],
            options={"temperature": 0.3},
            keep_alive=0,  # One-shot vision analysis - unload immediately to prevent VRAM thrashing
        )
        duration = _time.perf_counter() - start
        content = response.get("message", {}).get("content", "Could not analyze image")
        _record_perf(
            "ollama_vision", duration, success=True,
            model=VISION_MODEL,
            output_tokens=len(content) // 4,
        )
        return content
    except Exception as e:
        duration = _time.perf_counter() - start
        _record_perf(
            "ollama_vision", duration, success=False,
            model=VISION_MODEL, error=str(e)[:200],
        )
        log(f"Vision model error: {e}")
        return f"Error: {e}"


def ask_claude_with_image(image_bytes: bytes, question: str) -> str:
    """
    Ask Claude to analyze an image via claude -p (Pro subscription).

    Saves image to a temp file, then prompts Claude Code to read and
    analyze it. Uses Pro subscription instead of API credits.
    Falls back to error message if claude -p fails.
    """
    import tempfile
    from pathlib import Path

    from .claude_code_runner import run_claude_prompt

    # Save image to temp file so Claude Code can read it
    suffix = ".png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        suffix = ".jpg"
    elif image_bytes[:4] == b"GIF8":
        suffix = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        suffix = ".webp"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="technomancer_img_"
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Build prompt that tells Claude Code to read and analyze the image
        prompt = (
            f"Read the image file at {tmp_path} and analyze it.\n\n"
            f"The user's question about this image: {question}\n\n"
            f"Provide a detailed, helpful analysis. Be specific about what "
            f"you see in the image."
        )

        vision_start = _time.perf_counter()
        result = run_claude_prompt(prompt, timeout=60, max_turns=3)
        vision_duration = _time.perf_counter() - vision_start

        if result["success"]:
            _record_perf(
                "claude_pro_sub", vision_duration, success=True,
                model="claude-code-pro",
            )
            return result["result"]
        else:
            _record_perf(
                "claude_pro_sub", vision_duration, success=False,
                model="claude-code-pro",
                error=str(result.get("error", ""))[:200],
            )
            log(f"claude -p image analysis failed: {result.get('error')}")
            return f"Error analyzing image: {result.get('error')}"

    except Exception as e:
        log(f"Image analysis error: {e}")
        return f"Error analyzing image: {e}"

    finally:
        # Clean up temp file
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def identify_image(
    image_bytes: bytes, user_question: str = "Who is this?"
) -> ImageIdentificationResult:
    """
    Cascade through identification methods:
    1. Local vision model (llava-llama3)
    2. Claude (fallback)

    Returns dict with:
        - method: which method succeeded
        - result: the identification result
        - confidence: how confident the result is
    """
    log("Starting cascading image identification...")

    # Step 1: Try local vision model
    log("Step 1: Trying local vision model (llava-llama3)...")
    vision_result = analyze_with_vision_model([image_bytes])

    # Check if response actually identifies someone by name
    # Generic descriptions like "a woman with dark hair" don't count
    result_lower = vision_result.lower()

    # Signs that the model gave a generic description instead of identification
    generic_description_signs = [
        "i cannot identify",
        "i don't recognize",
        "unclear",
        "cannot determine",
        "aren't enough distinctive features",
        "without more context",
        "not enough information",
        "i can see it's an illustrated",
        "i can see it's a",
        "appears to be a young",
        "appears to be an",
        "a woman with",
        "a man with",
        "a character with",
        "can't identify exactly",
        "don't have enough",
        "hard to identify",
        "difficult to determine",
    ]

    gave_generic_description = any(sign in result_lower for sign in generic_description_signs)

    # Check if it actually named a specific character/person
    # Look for patterns like "This is [Name]" or "[Name] from [Series]"
    import re

    has_specific_name = bool(
        re.search(r"\b(this is|that\'s|it\'s|she is|he is)\s+[A-Z][a-z]+", vision_result)
    )
    _has_from_series = bool(re.search(r"from\s+(the\s+)?[A-Z][a-z]+", vision_result))

    # Only accept vision model result if it actually identified someone
    if has_specific_name and not gave_generic_description:
        log(f"Vision model identified: {vision_result[:100]}...")
        return {
            "method": "vision_model",
            "result": vision_result,
            "confidence": 0.7,
        }

    log("Vision model gave description but no identification, falling back to Claude...")

    # Step 2: Fall back to Claude
    log("Step 2: Asking Claude...")
    claude_prompt = f"""Look at this image and answer: {user_question}

If this is an anime/game character, identify them specifically (name and series).
If you're not sure, describe what you see and make your best guess."""

    claude_result = ask_claude_with_image(image_bytes, claude_prompt)

    if "error" not in claude_result.lower() and "not found" not in claude_result.lower():
        log(f"Claude response: {claude_result[:100]}...")
        return {
            "method": "claude",
            "result": claude_result,
            "confidence": 0.9,
        }

    # If Claude also failed, return the vision model result anyway
    log("All methods attempted, returning best available result")
    return {
        "method": "vision_model",
        "result": vision_result,
        "confidence": 0.5,
    }
