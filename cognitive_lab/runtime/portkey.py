from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
DEFAULT_PROVIDER = "@topicosdeengenhariadesoftwarre"
DEFAULT_MODEL = "z-ai/glm-4.5-air:free"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


@dataclass(slots=True)
class PortkeyLangGraphConfig:
    api_key: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    base_url: str = PORTKEY_GATEWAY_URL or "https://api.portkey.ai/v1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_message: str = "Hello chatgpt, how are you?"
    thread_id: str = "default"
    trace_id: str | None = None
    temperature: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "PortkeyLangGraphConfig":
        load_dotenv()

        api_key = os.getenv("PORTKEY_API_KEY")
        if not api_key:
            raise ValueError(
                "PORTKEY_API_KEY is required. Add it to your .env file before running the graph."
            )

        provider = os.getenv("PORTKEY_PROVIDER", DEFAULT_PROVIDER)
        model = os.getenv("PORTKEY_MODEL", DEFAULT_MODEL)
        base_url = os.getenv("PORTKEY_BASE_URL", PORTKEY_GATEWAY_URL or "https://api.portkey.ai/v1")
        system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        user_message = os.getenv("USER_MESSAGE", "Hello chatgpt, how are you?")
        thread_id = os.getenv("LANGGRAPH_THREAD_ID", "default")
        trace_id = os.getenv("PORTKEY_TRACE_ID") or f"langgraph-{thread_id}-{uuid4().hex[:8]}"
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))

        metadata = {
            "framework": "langgraph",
            "entrypoint": "llm_call.py",
            "graph_id": "chatbot_v1",
        }

        app_env = os.getenv("APP_ENV")
        if app_env:
            metadata["app_env"] = app_env

        user_id = os.getenv("PORTKEY_USER_ID")
        if user_id:
            metadata["_user"] = user_id

        return cls(
            api_key=api_key,
            provider=provider,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            user_message=user_message,
            thread_id=thread_id,
            trace_id=trace_id,
            temperature=temperature,
            metadata=metadata,
        )


def build_chat_model(config: PortkeyLangGraphConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        api_key="dummy",
        base_url=config.base_url,
        temperature=config.temperature,
        default_headers=createHeaders(
            api_key=config.api_key,
            provider=config.provider,
            trace_id=config.trace_id,
            metadata=config.metadata,
        ),
    )


def build_chat_graph(config: PortkeyLangGraphConfig):
    llm = build_chat_model(config)
    graph_builder = StateGraph(ChatState)

    def chatbot(state: ChatState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


def build_initial_messages(config: PortkeyLangGraphConfig) -> list[BaseMessage]:
    return [
        SystemMessage(content=config.system_prompt),
        HumanMessage(content=config.user_message),
    ]


def invoke_graph_once(config: PortkeyLangGraphConfig) -> dict[str, Any]:
    graph = build_chat_graph(config)
    result = graph.invoke(
        {"messages": build_initial_messages(config)},
        config={"configurable": {"thread_id": config.thread_id}},
    )

    final_message = _extract_last_ai_message(result["messages"])

    return {
        "trace_id": config.trace_id,
        "thread_id": config.thread_id,
        "provider": config.provider,
        "model": config.model,
        "response": final_message.content,
    }


def _extract_last_ai_message(messages: list[BaseMessage]) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message

    raise ValueError("The graph finished without producing an AI message.")
