import requests
from bs4 import BeautifulSoup
from openai.types.chat.chat_completion_chunk import ChoiceDelta
import json
import asyncio

def get_url_content(url: str) -> str:
    return "\n".join(
        line 
        for line in BeautifulSoup(requests.get(url).text, "html.parser")\
            .get_text()\
                .splitlines() 
        if line.strip()
    )


def handle_tool_call(message: ChoiceDelta) -> tuple[dict, str]:
    tool_call = message.tool_calls[0]  # type: ignore
    function_name = tool_call.function.name #type: ignore
    try:
        arguments = json.loads(tool_call.function.arguments)  # type: ignore
    except json.JSONDecodeError:
        response = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"error": "Invalid JSON in tool arguments"})
        }
        return response, ""

    match function_name:
        case "get_url_content":
            url = arguments.get("url")
            content = asyncio.run(asyncio.to_thread(get_url_content, url))
            payload = {"url": url, "content": content}
        case _:
            payload = {"error": f"Unknown tool: {function_name}"}

    response = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(payload)
    }
    return response, arguments.get("url", "")
