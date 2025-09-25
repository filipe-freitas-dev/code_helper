import os
import anthropic
from openai import OpenAI
import gradio as gr
from typing import Iterable, Generator, cast
from dotenv import load_dotenv
from anthropic.types import MessageParam
from openai.types.chat import ChatCompletionMessageParam
from abc import ABC, abstractmethod
from tools import handle_tool_call
from enum import Enum


class ApiKeyError(Exception):...

class ChatBot(ABC):
    def __init__(self, system_prompt: str, context: str | None = None) -> None:
        load_dotenv(override=True)
        self.context_loaded = False
        self.system_prompt = system_prompt
        self.context = context

    @abstractmethod
    def chat(self, message: str, history: Iterable) -> Generator:
        pass

    def run(self) -> None:
        gr.ChatInterface(fn=self.chat, type="messages").launch()


class OpenAiChatBot(ChatBot):
    std_system_prompt = (
        "You are a helpful assistant. "
        "If you don't know the answer, say so. "
        "Don't try to make up an answer if you don't know. "
        "Make questions to the user to try to help more your accuracy."
        )
    
    def __init__(self, system_prompt: str = std_system_prompt, context: str | None = None) -> None:
        super().__init__(system_prompt, context)
        if not os.getenv("OPENAI_API_KEY"):
            raise ApiKeyError("API Key not found.")
        self.model = OpenAI()


    def chat(self, message: str, history: Iterable[ChatCompletionMessageParam]) -> Generator[str, None, None]:
        if not self.context_loaded and self.context:
            message = f"\n\nContext: {self.context}\n\n{message}"
            self.context_loaded = True

        history_list = list(history)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(history_list)
        messages.append({"role": "user", "content": message})
        try:
            first = self.model.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=False,
                tools=define_tools()
            )
        except Exception as e:
            yield f"Error calling model: {e}"
            return

        if not first.choices:
            yield "Model returned no choices."
            return

        first_msg = first.choices[0].message

       
        if getattr(first_msg, "tool_calls", None):
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "assistant",
                "content": first_msg.content,
                "tool_calls": first_msg.tool_calls,
            }))

            try:
                tool_response, _ = handle_tool_call(first_msg)  
            except Exception as e:
                tool_response = {
                    "role": "tool",
                    "tool_call_id": "",
                    "content": f"{{\"error\": \"Tool execution failed: {e}\"}}"
                }

            messages.append(cast(ChatCompletionMessageParam, tool_response))

        try:
            stream = self.model.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=True
            )
        except Exception as e:
            yield f"Error calling model for streaming response: {e}"
            return

        response = ""
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                response += (getattr(delta, "content", None) or "")
            except Exception:
                continue
            yield response


class AnthropicChatBot(ChatBot):
    std_system_prompt = (
        "You are a helpful assistant. "
        "If you don't know the answer, say so. "
        "Don't try to make up an answer if you don't know. "
        "Make questions to the user to try to help more your accuracy."
        )
    
    def __init__(self, system_prompt: str = std_system_prompt, context: str | None = None) -> None:
        super().__init__(system_prompt, context)
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ApiKeyError("API Key not found.")
        self.model = anthropic.Anthropic()

    def chat(self, message: str, history:Iterable[MessageParam]) -> Generator[str]:
        if not self.context_loaded and self.context:
            context_with_label = "\n\nContext: " + self.context + "\n\n"
            message = context_with_label + message
            self.context_loaded = True

        messages: list[MessageParam] = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": message})

        result = self.model.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=5489,
            temperature=0.7,
            system=self.system_prompt,
            messages=messages,
            tools=define_tools("anthropic")
        )


        with result as stream:
            new_text = ""
            for text in stream.text_stream:
                new_text += text
                yield new_text
    

class OpenAiFunctionTools(Enum):
    get_url_function_tool = {
        "name": "get_url_content",
        "description": "Find the content about of urls that the user gives. Call this whenever you need to know some content about some url.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url that the user wants to know something about",
                },
            },
            "required": ["url"],
            "additionalProperties": False
        }
    }

    google_search_tool = {
        "name": "google_tool_recursive",
        "description": "Use this tool to search Google only when the user explicitly says 'pesquise'. Perform the search to gather information that supports and informs your answer. Do not use it for other purposes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use in Google",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 3
                },
                "depth": {
                    "type": "integer",
                    "description": "Depth of recursive search",
                    "default": 2
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }

class AnthropicFunctionTools(Enum):
    get_url_function_tool = {
        "name": "get_url_content",
        "description": "Extract text content from a given URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract content from"
                }
            },
            "required": ["url"]
        }
    }

def define_tools(model: str = "o") -> list:
    if model.lower().startswith("a"):
        return [{"type":"function","function": function.value} for function in AnthropicFunctionTools]

    return [{"type":"function","function": function.value} for function in OpenAiFunctionTools]
