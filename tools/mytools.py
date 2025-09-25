import requests
from bs4 import BeautifulSoup
from openai.types.chat.chat_completion_chunk import ChoiceDelta
import json

def get_url_content(url: str) -> str:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "code_helper/1.0 (+https://localhost)"
                },
            )
            resp.raise_for_status()
            text = resp.text
            return "\n".join(
                line
                for line in BeautifulSoup(text, "html.parser")
                    .get_text()
                    .splitlines()
                if line.strip()
            )
        except Exception as e:
            last_err = e
            continue
        
    raise RuntimeError(f"Failed to fetch content from URL after retries: {last_err}")


def handle_tool_call(message: ChoiceDelta) -> tuple[dict, str]:
    if not getattr(message, "tool_calls", None):  # type: ignore[attr-defined]
        return {
            "role": "tool",
            "tool_call_id": "",
            "content": json.dumps({"error": "No tool_calls present in message delta"})
        }, ""

    tool_call = message.tool_calls[0]  # type: ignore[index]
    function_name = getattr(tool_call.function, "name", None)  # type: ignore[attr-defined]

    if not function_name:
        return {
            "role": "tool",
            "tool_call_id": getattr(tool_call, "id", ""),
            "content": json.dumps({"error": "Tool function name missing"})
        }, ""

    raw_args = getattr(tool_call.function, "arguments", "{}")  # type: ignore[attr-defined]
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        response = {
            "role": "tool",
            "tool_call_id": getattr(tool_call, "id", ""),
            "content": json.dumps({"error": "Invalid JSON in tool arguments", "raw": raw_args})
        }
        return response, ""

    match function_name:
        case "get_url_content":
            url = arguments.get("url") if isinstance(arguments, dict) else None
            if not url or not isinstance(url, str):
                payload = {"error": "Missing or invalid 'url' parameter"}
            else:
                try:
                    content = get_url_content(url)
                    payload = {"url": url, "content": content}
                except Exception as e:
                    payload = {"error": f"Failed to fetch URL: {e}", "url": url}
        case _:
            payload = {"error": f"Unknown tool: {function_name}"}

    response = {
        "role": "tool",
        "tool_call_id": getattr(tool_call, "id", ""),
        "content": json.dumps(payload)
    }
    return response, arguments.get("url", "") if isinstance(arguments, dict) else ""
