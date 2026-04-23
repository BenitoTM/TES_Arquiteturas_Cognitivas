from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cognitive_lab.agents import react_coala, reflection
from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model


DEFAULT_GRAPH_MERMAID = "artifacts/graphs/reflection_langgraph_graph.mmd"


class ReflectionLangGraphState(TypedDict, total=False):
    question: str
    max_attempts: int
    max_steps: int
    attempt_number: int
    step: int
    memory_dir: str
    working_memory: dict[str, Any]
    reflections: list[dict[str, Any]]
    semantic_hits: list[dict[str, Any]]
    episodic_hits: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    planner_kind: str
    planner_text: str
    thought: str | None
    action_name: str | None
    action_input: str | None
    observation: str | None
    final_answer: str | None
    judge_verdict: str | None
    judge_feedback: str | None
    latest_reflection: str | None
    total_tokens: int
    llm_calls: int
    memory_counts: dict[str, int]


def build_reflection_langgraph(config: PortkeyLangGraphConfig):
    llm = build_chat_model(config)
    tools = react_coala.build_tool_registry()
    graph_builder = StateGraph(ReflectionLangGraphState)

    def bootstrap(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        memory_dir = Path(state["memory_dir"]).resolve()
        print("=== INICIANDO AGENTE REFLECTION (LANGGRAPH) ===")
        print(f"Pergunta: {state['question']}\n")
        print(f"Memoria persistente: {memory_dir}\n")
        return {
            "attempt_number": 1,
            "step": 0,
            "trajectory": [],
            "total_tokens": 0,
            "llm_calls": 0,
            "working_memory": {
                "goal": state["question"],
                "step": 0,
                "last_thought": None,
                "last_action": None,
                "last_observation": None,
            },
        }

    def recall_reflections(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        root = Path(state["memory_dir"])
        reflection_memory = reflection.ReflectionMemoryStore(root)
        memory = react_coala.CoALAMemoryStore(root)
        last_observation = state["working_memory"].get("last_observation") or ""
        query = f"{state['question']}\n{last_observation}"
        reflections = reflection_memory.search_reflections(query, top_k=3)
        semantic_hits = memory.search_semantic(query, top_k=3)
        episodic_hits = memory.search_episodic(query, top_k=3)
        return {
            "reflections": reflections,
            "semantic_hits": semantic_hits,
            "episodic_hits": episodic_hits,
            "memory_counts": {
                **memory.counts(),
                "reflections": reflection_memory.count(),
            },
        }

    def actor(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        prompt = reflection.build_reflection_attempt_prompt(
            question=state["question"],
            attempt_number=state["attempt_number"],
            max_attempts=state["max_attempts"],
            max_steps=state["max_steps"],
            working_memory=state["working_memory"],
            reflections=state.get("reflections", []),
            semantic_hits=state.get("semantic_hits", []),
            episodic_hits=state.get("episodic_hits", []),
            trajectory=state.get("trajectory", []),
        )
        response = llm.invoke(
            [
                react_coala.SystemMessage(content=reflection.build_reflection_actor_system_prompt(tools)),
                react_coala.HumanMessage(content=prompt),
            ],
            stop=["Observation:"],
        )
        tokens = state.get("total_tokens", 0)
        llm_calls = state.get("llm_calls", 0) + 1
        if getattr(response, "usage_metadata", None):
            tokens += response.usage_metadata.get("total_tokens", 0)

        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = react_coala._parse_react_output(text)
        current_step = state.get("step", 0) + 1

        print(f"--- Tentativa {state['attempt_number']} | Passo {current_step} ---")
        print(text.strip() or "<vazio>")

        updated_working_memory = dict(state["working_memory"])
        updated_working_memory["step"] = current_step

        result: ReflectionLangGraphState = {
            "step": current_step,
            "planner_text": text,
            "total_tokens": tokens,
            "llm_calls": llm_calls,
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

    def execute_action(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        action_name = state.get("action_name")
        action_input = state.get("action_input") or ""
        if not action_name or action_name not in tools:
            return {
                "observation": f"Ferramenta invalida: {action_name}. Ferramentas validas: {', '.join(sorted(tools))}."
            }

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

    def record_step(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
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
        return {"working_memory": updated_working_memory, "trajectory": trajectory}

    def judge(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        if not state.get("final_answer"):
            feedback = "A tentativa terminou sem produzir uma resposta final clara."
            print("Judge Verdict: RETRY")
            print(f"Judge Feedback: {feedback}")
            return {"judge_verdict": "RETRY", "judge_feedback": feedback}

        if react_coala._normalize_text(state["question"]) == react_coala._normalize_text(react_coala.OFFICIAL_BENCHMARK_QUESTION):
            evaluation = react_coala.evaluate_official_benchmark_answer(state["final_answer"])
            verdict = "ACCEPT" if evaluation["correct"] else "RETRY"
            print(f"Judge Verdict: {verdict}")
            print(f"Judge Feedback: {evaluation['feedback']}")
            return {
                "judge_verdict": verdict,
                "judge_feedback": evaluation["feedback"],
            }

        response = llm.invoke(
            [
                react_coala.SystemMessage(content="Voce avalia respostas de agentes com rigor e objetividade."),
                react_coala.HumanMessage(
                    content=reflection.build_reflection_judge_prompt(
                        question=state["question"],
                        final_answer=state.get("final_answer"),
                        trajectory=state.get("trajectory", []),
                    )
                ),
            ]
        )
        tokens = state.get("total_tokens", 0)
        llm_calls = state.get("llm_calls", 0) + 1
        if getattr(response, "usage_metadata", None):
            tokens += response.usage_metadata.get("total_tokens", 0)

        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = reflection.parse_judge_output(text)
        print(f"Judge Verdict: {parsed['verdict']}")
        print(f"Judge Feedback: {parsed['feedback']}")
        return {
            "judge_verdict": parsed["verdict"],
            "judge_feedback": parsed["feedback"],
            "total_tokens": tokens,
            "llm_calls": llm_calls,
        }

    def reflect(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        root = Path(state["memory_dir"])
        reflection_memory = reflection.ReflectionMemoryStore(root)
        response = llm.invoke(
            [
                react_coala.SystemMessage(content="Voce escreve reflexoes curtas e acionaveis para melhorar a proxima tentativa."),
                react_coala.HumanMessage(
                    content=reflection.build_reflection_prompt(
                        question=state["question"],
                        attempt_result={
                            "final_answer": state.get("final_answer"),
                            "trajectory": state.get("trajectory", []),
                        },
                        feedback=state.get("judge_feedback") or "Sem feedback.",
                    )
                ),
            ]
        )
        tokens = state.get("total_tokens", 0)
        llm_calls = state.get("llm_calls", 0) + 1
        if getattr(response, "usage_metadata", None):
            tokens += response.usage_metadata.get("total_tokens", 0)

        reflection_text = reflection.parse_reflection_output(
            response.content if isinstance(response.content, str) else str(response.content)
        )
        reflection_memory.add_reflection(
            question=state["question"],
            reflection=reflection_text,
            feedback=state.get("judge_feedback") or "Sem feedback.",
            trajectory_summary=react_coala._summarize_trajectory(state.get("trajectory", [])),
        )
        print(f"Reflection: {reflection_text}\n")
        return {
            "latest_reflection": reflection_text,
            "total_tokens": tokens,
            "llm_calls": llm_calls,
            "attempt_number": state["attempt_number"] + 1,
            "step": 0,
            "trajectory": [],
            "final_answer": None,
            "judge_verdict": None,
            "judge_feedback": None,
            "working_memory": {
                "goal": state["question"],
                "step": 0,
                "last_thought": None,
                "last_action": None,
                "last_observation": None,
            },
        }

    def finalize(state: ReflectionLangGraphState) -> ReflectionLangGraphState:
        root = Path(state["memory_dir"])
        memory = react_coala.CoALAMemoryStore(root)
        reflection_memory = reflection.ReflectionMemoryStore(root)

        final_answer = state.get("final_answer")
        if state.get("judge_verdict") == "ACCEPT" and final_answer:
            # Persistencia da tentativa aceita para memoria episodica e semantica.
            memory.add_episode(
                state["question"],
                final_answer,
                react_coala._summarize_trajectory(state.get("trajectory", [])),
            )
            react_coala._auto_consolidate_semantic_memory(memory, state["question"], final_answer)
            print("\n=== RESPOSTA FINAL ACEITA ===")
        else:
            final_answer = "Nao foi possivel obter uma resposta aceita dentro do limite de tentativas."
            memory.add_episode(
                state["question"],
                final_answer,
                react_coala._summarize_trajectory(state.get("trajectory", [])),
            )
            print("\n=== LIMITE DE TENTATIVAS ATINGIDO ===")

        return {
            "final_answer": final_answer,
            "memory_counts": {
                **memory.counts(),
                "reflections": reflection_memory.count(),
            },
        }

    def actor_route(state: ReflectionLangGraphState) -> Literal["execute_action", "record_step", "judge"]:
        if state.get("planner_kind") == "final":
            return "judge"
        if state.get("planner_kind") == "action":
            return "execute_action"
        return "record_step"

    def record_route(state: ReflectionLangGraphState) -> Literal["actor", "judge"]:
        if state["step"] >= state["max_steps"]:
            return "judge"
        return "actor"

    def judge_route(state: ReflectionLangGraphState) -> Literal["finalize", "reflect"]:
        if state.get("judge_verdict") == "ACCEPT":
            return "finalize"
        if state["attempt_number"] >= state["max_attempts"]:
            return "finalize"
        return "reflect"

    # O grafo implementa o ciclo Reflection: agir -> avaliar -> refletir -> tentar novamente.
    graph_builder.add_node("bootstrap", bootstrap)
    graph_builder.add_node("recall_reflections", recall_reflections)
    graph_builder.add_node("actor", actor)
    graph_builder.add_node("execute_action", execute_action)
    graph_builder.add_node("record_step", record_step)
    graph_builder.add_node("judge", judge)
    graph_builder.add_node("reflect", reflect)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "bootstrap")
    graph_builder.add_edge("bootstrap", "recall_reflections")
    graph_builder.add_edge("recall_reflections", "actor")
    graph_builder.add_conditional_edges(
        "actor",
        actor_route,
        {
            "execute_action": "execute_action",
            "record_step": "record_step",
            "judge": "judge",
        },
    )
    graph_builder.add_edge("execute_action", "record_step")
    graph_builder.add_conditional_edges(
        "record_step",
        record_route,
        {
            "actor": "actor",
            "judge": "judge",
        },
    )
    # Depois de cada tentativa, o juiz decide se encerramos ou se precisamos refletir e repetir.
    graph_builder.add_conditional_edges(
        "judge",
        judge_route,
        {
            "finalize": "finalize",
            "reflect": "reflect",
        },
    )
    graph_builder.add_edge("reflect", "recall_reflections")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=MemorySaver())


def invoke_reflection_langgraph_once(
    config: PortkeyLangGraphConfig,
    question: str,
    max_attempts: int = 3,
    max_steps: int = 6,
    memory_dir: str | Path = reflection.DEFAULT_REFLECTION_MEMORY_DIR,
    graph: Any | None = None,
) -> dict[str, Any]:
    graph = graph or build_reflection_langgraph(config)
    started_at = time.perf_counter()
    result = graph.invoke(
        {
            "question": question,
            "max_attempts": max_attempts,
            "max_steps": max_steps,
            "memory_dir": str(memory_dir),
        },
        config={"configurable": {"thread_id": f"{config.thread_id}-reflection-langgraph"}},
    )
    return {
        "trace_id": config.trace_id,
        "thread_id": f"{config.thread_id}-reflection-langgraph",
        "provider": config.provider,
        "model": config.model,
        "resposta": result["final_answer"],
        "attempts": result["attempt_number"],
        "steps": result["step"],
        "tokens": result["total_tokens"],
        "llm_calls": result.get("llm_calls", 0),
        "total_time_seconds": round(time.perf_counter() - started_at, 4),
        "trajectory": result.get("trajectory", []),
        "judge_feedback": result.get("judge_feedback"),
        "memory_dir": str(Path(memory_dir).resolve()),
        "memory_counts": result.get("memory_counts", {}),
    }


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv(
        "REFLECTION_USER_MESSAGE",
        os.getenv("REACT_USER_MESSAGE", config.user_message or react_coala.OFFICIAL_BENCHMARK_QUESTION),
    )
    max_attempts = int(os.getenv("REFLECTION_MAX_ATTEMPTS", "3"))
    max_steps = int(os.getenv("REFLECTION_MAX_STEPS", "6"))
    memory_dir = Path(os.getenv("REFLECTION_MEMORY_DIR", reflection.DEFAULT_REFLECTION_MEMORY_DIR))
    graph_mermaid = Path(os.getenv("REFLECTION_LANGGRAPH_MERMAID", DEFAULT_GRAPH_MERMAID))
    graph_png_value = os.getenv("REFLECTION_LANGGRAPH_PNG", "").strip()

    graph = build_reflection_langgraph(config)
    graph_mermaid.parent.mkdir(parents=True, exist_ok=True)
    graph_mermaid.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    graph_message = f"Mermaid salvo em: {graph_mermaid.resolve()}"
    if graph_png_value:
        graph_png = Path(graph_png_value)
        try:
            graph_png.parent.mkdir(parents=True, exist_ok=True)
            graph_png.write_bytes(graph.get_graph().draw_mermaid_png())
            graph_message += f"\nPNG salvo em: {graph_png.resolve()}"
        except Exception as exc:
            graph_message += f"\nNao foi possivel gerar o PNG do grafo: {exc}"

    result = invoke_reflection_langgraph_once(
        config=config,
        question=question,
        max_attempts=max_attempts,
        max_steps=max_steps,
        memory_dir=memory_dir,
        graph=graph,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{graph_message}")


if __name__ == "__main__":
    main()
