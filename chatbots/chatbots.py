import os
import anthropic
from openai import OpenAI
import gradio as gr
from typing import Iterable, Generator
from dotenv import load_dotenv
from anthropic.types import MessageParam
from openai.types.chat import ChatCompletionMessageParam
from abc import ABC, abstractmethod

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
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history_list)
        messages.append({"role": "user", "content": message})

        stream = self.model.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ""
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

        messages: Iterable[MessageParam] = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": message})

        result = self.model.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=5489,
            temperature=0.7,
            system=self.system_prompt,
            messages=messages,
        )

        with result as stream:
            new_text = ""
            for text in stream.text_stream:
                new_text += text
                yield new_text
    