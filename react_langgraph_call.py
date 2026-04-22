from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents import react_coala


DEFAULT_GRAPH_MERMAID = "artifacts/graphs/react_langgraph_graph.mmd"


class ReactLangGraphState(TypedDict, total=False):
    question: str
    max_steps: int
    step: int
    memory_dir: str
    working_memory: dict[str, Any]
    semantic_hits: list[dict[str, Any]]
    episodic_hits: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    planner_text: str
    planner_kind: str
    thought: str | None
    action_name: str | None
    action_input: str | None
    observation: str | None
    final_answer: str | None
    total_tokens: int
    memory_counts: dict[str, int]


def build_react_langgraph(config: PortkeyLangGraphConfig):
    llm = build_chat_model(config)
    tools = react_coala.build_tool_registry()
    graph_builder = StateGraph(ReactLangGraphState)

    def bootstrap(state: ReactLangGraphState) -> ReactLangGraphState:
        memory_dir = Path(state["memory_dir"]).resolve()
        print("=== INICIANDO AGENTE REACT + COALA (LANGGRAPH) ===")
        print(f"Pergunta: {state['question']}\n")
        print(f"Memoria persistente: {memory_dir}\n")
        return {
            "step": 0,
            "trajectory": [],
            "total_tokens": 0,
            "working_memory": {
                "goal": state["question"],
                "step": 0,
                "last_thought": None,
                "last_action": None,
                "last_observation": None,
            },
        }

    def recall_memories(state: ReactLangGraphState) -> ReactLangGraphState:
        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        last_observation = state["working_memory"].get("last_observation") or ""
        query = f"{state['question']}\n{last_observation}"
        semantic_hits = memory.search_semantic(query, top_k=3)
        episodic_hits = memory.search_episodic(query, top_k=3)
        return {
            "semantic_hits": semantic_hits,
            "episodic_hits": episodic_hits,
            "memory_counts": memory.counts(),
        }

    def planner(state: ReactLangGraphState) -> ReactLangGraphState:
        prompt = react_coala._build_decision_prompt(
            question=state["question"],
            working_memory=state["working_memory"],
            semantic_hits=state.get("semantic_hits", []),
            episodic_hits=state.get("episodic_hits", []),
            trajectory=state.get("trajectory", []),
            max_steps=state["max_steps"],
        )

        response = llm.invoke(
            [
                react_coala.SystemMessage(content=react_coala.build_react_system_prompt(tools)),
                react_coala.HumanMessage(content=prompt),
            ],
            stop=["Observation:"],
        )

        tokens = state.get("total_tokens", 0)
        if getattr(response, "usage_metadata", None):
            tokens += response.usage_metadata.get("total_tokens", 0)

        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = react_coala._parse_react_output(text)
        current_step = state.get("step", 0) + 1

        print(f"--- Passo {current_step} ---")
        print(text.strip() or "<vazio>")

        updated_working_memory = dict(state["working_memory"])
        updated_working_memory["step"] = current_step

        result: ReactLangGraphState = {
            "step": current_step,
            "planner_text": text,
            "total_tokens": tokens,
            "working_memory": updated_working_memory,
            "action_name": None,
            "action_input": None,
            "observation": None,
            "final_answer": None,
        }

        if parsed["kind"] == "final":
            result["planner_kind"] = "final"
            result["final_answer"] = parsed["final_answer"]
            return result

        if parsed["kind"] == "error":
            updated_working_memory["last_thought"] = "Formato incorreto."
            updated_working_memory["last_action"] = None
            result["planner_kind"] = "error"
            result["thought"] = "Formato incorreto."
            result["observation"] = parsed["error"]
            return result

        updated_working_memory["last_thought"] = parsed["thought"]
        updated_working_memory["last_action"] = f"{parsed['action']}[{parsed['action_input']}]"
        result["planner_kind"] = "action"
        result["thought"] = parsed["thought"]
        result["action_name"] = parsed["action"]
        result["action_input"] = parsed["action_input"]
        return result

    def execute_action(state: ReactLangGraphState) -> ReactLangGraphState:
        action_name = state.get("action_name")
        action_input = state.get("action_input") or ""

        if not action_name or action_name not in tools:
            observation = (
                f"Ferramenta invalida: {action_name}. "
                f"Ferramentas validas: {', '.join(sorted(tools))}."
            )
            return {"observation": observation}

        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        runtime = react_coala.ToolRuntime(
            memory=memory,
            question=state["question"],
            working_memory=state["working_memory"],
            trajectory=state.get("trajectory", []),
        )

        try:
            observation = tools[action_name].handler(action_input, runtime)
        except Exception as exc:
            observation = f"Erro ao executar ferramenta {action_name}: {exc}"

        return {"observation": observation}

    def record_step(state: ReactLangGraphState) -> ReactLangGraphState:
        updated_working_memory = dict(state["working_memory"])
        updated_working_memory["last_observation"] = state.get("observation")
        trajectory = list(state.get("trajectory", []))
        trajectory.append(
            {
                "step": state["step"],
                "thought": state.get("thought"),
                "action": state.get("action_name"),
                "action_input": state.get("action_input"),
                "observation": state.get("observation"),
            }
        )
        print(f"Observation: {state.get('observation')}")
        return {
            "working_memory": updated_working_memory,
            "trajectory": trajectory,
        }

    def finalize(state: ReactLangGraphState) -> ReactLangGraphState:
        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        final_answer = state.get("final_answer")
        if not final_answer:
            final_answer = "Limite de passos atingido antes de chegar a uma resposta final."

        memory.add_episode(
            state["question"],
            final_answer,
            react_coala._summarize_trajectory(state.get("trajectory", [])),
        )
        react_coala._auto_consolidate_semantic_memory(memory, state["question"], final_answer)

        print("\n=== RESPOSTA FINAL ENCONTRADA ===")
        return {
            "final_answer": final_answer,
            "memory_counts": memory.counts(),
        }

    def planner_route(state: ReactLangGraphState) -> Literal["execute_action", "record_step", "finalize"]:
        if state.get("planner_kind") == "final":
            return "finalize"
        if state.get("planner_kind") == "action":
            return "execute_action"
        return "record_step"

    def record_route(state: ReactLangGraphState) -> Literal["recall_memories", "finalize"]:
        if state["step"] >= state["max_steps"]:
            return "finalize"
        return "recall_memories"

    graph_builder.add_node("bootstrap", bootstrap)
    graph_builder.add_node("recall_memories", recall_memories)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("execute_action", execute_action)
    graph_builder.add_node("record_step", record_step)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "bootstrap")
    graph_builder.add_edge("bootstrap", "recall_memories")
    graph_builder.add_edge("recall_memories", "planner")
    graph_builder.add_conditional_edges(
        "planner",
        planner_route,
        {
            "execute_action": "execute_action",
            "record_step": "record_step",
            "finalize": "finalize",
        },
    )
    graph_builder.add_edge("execute_action", "record_step")
    graph_builder.add_conditional_edges(
        "record_step",
        record_route,
        {
            "recall_memories": "recall_memories",
            "finalize": "finalize",
        },
    )
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=MemorySaver())


def invoke_react_langgraph_once(
    config: PortkeyLangGraphConfig,
    question: str,
    max_steps: int = 10,
    memory_dir: str | Path = react_coala.DEFAULT_MEMORY_DIR,
    graph: Any | None = None,
) -> dict[str, Any]:
    graph = graph or build_react_langgraph(config)
    result = graph.invoke(
        {
            "question": question,
            "max_steps": max_steps,
            "memory_dir": str(memory_dir),
        },
        config={"configurable": {"thread_id": f"{config.thread_id}-react-langgraph"}},
    )

    return {
        "trace_id": config.trace_id,
        "thread_id": f"{config.thread_id}-react-langgraph",
        "provider": config.provider,
        "model": config.model,
        "resposta": result["final_answer"],
        "steps": result["step"],
        "tokens": result["total_tokens"],
        "trajectory": result.get("trajectory", []),
        "memory_dir": str(Path(memory_dir).resolve()),
        "memory_counts": result.get("memory_counts", {}),
    }


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv("REACT_USER_MESSAGE", config.user_message)
    max_steps = int(os.getenv("REACT_MAX_STEPS", "10"))
    memory_dir = Path(os.getenv("COALA_MEMORY_DIR", react_coala.DEFAULT_MEMORY_DIR))
    graph_mermaid = Path(os.getenv("REACT_LANGGRAPH_MERMAID", DEFAULT_GRAPH_MERMAID))
    graph_png_value = os.getenv("REACT_LANGGRAPH_PNG", "").strip()

    graph = build_react_langgraph(config)
    graph_mermaid.parent.mkdir(parents=True, exist_ok=True)
    graph_mermaid.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    graph_message = f"Mermaid salvo em: {graph_mermaid.resolve()}"
    if graph_png_value:
        graph_png = Path(graph_png_value)
        try:
            graph_png.parent.mkdir(parents=True, exist_ok=True)
            png_bytes = graph.get_graph().draw_mermaid_png()
            graph_png.write_bytes(png_bytes)
            graph_message += f"\nPNG salvo em: {graph_png.resolve()}"
        except Exception as exc:
            graph_message += f"\nNao foi possivel gerar o PNG do grafo: {exc}"

    result = invoke_react_langgraph_once(
        config=config,
        question=question,
        max_steps=max_steps,
        memory_dir=memory_dir,
        graph=graph,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{graph_message}")


if __name__ == "__main__":
    main()
