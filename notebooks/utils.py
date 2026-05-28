"""Shared marimo display helpers for all ant-ai tutorial notebooks."""
from __future__ import annotations
import json as _json
import marimo as mo


def _event_element(event):
    if event.kind == "tool_calling" and getattr(event, "message", None):
        parts = []
        for tc in event.message.tool_calls:
            args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            parts.append(
                mo.callout(mo.md(f"**tool call** `{tc.function.name}({args_str})`"), kind="neutral")
            )
        return mo.vstack(parts) if parts else None
    if event.kind == "tool_calling" and event.content:
        return mo.callout(mo.md(f"**tool call** `{event.content}`"), kind="neutral")
    if event.kind == "tool_result" and event.content:
        return mo.accordion({"tool result": mo.md(f"```\n{event.content}\n```")})
    if event.kind == "final_answer" and event.content:
        return mo.callout(mo.md(event.content), kind="success")
    if event.content:
        return mo.md(f"_{event.content}_")
    return None


async def stream_output(aiter, prompt: str) -> None:
    """Stream agent events from `aiter` into the calling cell's marimo output."""
    items = [mo.callout(mo.md(f"**User:** {prompt}"), kind="info")]
    mo.output.replace(mo.vstack(items))
    async for event in aiter:
        el = _event_element(event)
        if el is not None:
            items.append(el)
            mo.output.replace(mo.vstack(items))


def show_raw_response(response: dict) -> None:
    """Display the full JSON-RPC response in a collapsible accordion (useful for inspecting structure)."""
    mo.output.replace(
        mo.accordion({"response": mo.md(f"```json\n{_json.dumps(response, indent=2)}\n```")})
    )


def show_response(response: dict) -> None:
    """Display an A2A response — extracts the agent answer or falls back to raw JSON."""
    try:
        answer = response["result"]["task"]["history"][-1]["parts"][0]["text"]
        mo.output.replace(mo.callout(mo.md(answer), kind="success"))
    except (KeyError, IndexError):
        mo.output.replace(
            mo.accordion({"raw response": mo.md(f"```json\n{_json.dumps(response, indent=2)}\n```")})
        )
