from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cognitive_lab.agents import lats, react_coala
from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model


DEFAULT_GRAPH_MERMAID = "artifacts/graphs/lats_langgraph_graph.mmd"


class LATSLangGraphState(TypedDict, total=False):
    question: str
    max_iterations: int
    branching_factor: int
    max_depth: int
    exploration_weight: float
    memory_dir: str
    iteration: int
    tree_nodes: dict[str, dict[str, Any]]
    selected_node_id: str | None
    semantic_hits: list[dict[str, Any]]
    episodic_hits: list[dict[str, Any]]
    expanded_child_ids: list[str]
    best_node_id: str | None
    final_answer: str | None
    total_tokens: int
    llm_calls: int
    memory_counts: dict[str, int]
    trajectory: list[dict[str, Any]]
    steps: int
    tree_size: int


def _deserialize_tree(tree_nodes: dict[str, dict[str, Any]]) -> dict[str, lats.SearchNode]:
    return {node_id: lats.deserialize_node(payload) for node_id, payload in tree_nodes.items()}


def _serialize_tree(nodes: dict[str, lats.SearchNode]) -> dict[str, dict[str, Any]]:
    return {node_id: lats.serialize_node(node) for node_id, node in nodes.items()}


def build_lats_langgraph(config: PortkeyLangGraphConfig):
    llm = build_chat_model(config)
    tools = react_coala.build_tool_registry()
    graph_builder = StateGraph(LATSLangGraphState)

    def bootstrap(state: LATSLangGraphState) -> LATSLangGraphState:
        memory_dir = Path(state["memory_dir"]).resolve()
        root = lats.create_root_node(state["question"])
        print("=== INICIANDO AGENTE LATS + COALA (LANGGRAPH) ===")
        print(f"Pergunta: {state['question']}\n")
        print(f"Memoria persistente: {memory_dir}\n")
        return {
            "iteration": 0,
            "tree_nodes": {root.node_id: lats.serialize_node(root)},
            "selected_node_id": None,
            "expanded_child_ids": [],
            "best_node_id": root.node_id,
            "final_answer": None,
            "total_tokens": 0,
            "llm_calls": 0,
            "trajectory": [],
            "steps": 0,
            "tree_size": 1,
        }

    def select_frontier(state: LATSLangGraphState) -> LATSLangGraphState:
        nodes = _deserialize_tree(state["tree_nodes"])
        selected = lats.select_frontier_node(
            nodes,
            max_depth=state["max_depth"],
            exploration_weight=state["exploration_weight"],
        )
        if selected is None:
            return {"selected_node_id": None}
        print(
            f"=== Iteracao {state['iteration'] + 1}/{state['max_iterations']} | no selecionado: {selected.node_id} "
            f"(profundidade={selected.depth}, score={selected.heuristic_score:.2f}, visitas={selected.visits}) ==="
        )
        return {"selected_node_id": selected.node_id}

    def recall_memories(state: LATSLangGraphState) -> LATSLangGraphState:
        if not state.get("selected_node_id"):
            return {}
        nodes = _deserialize_tree(state["tree_nodes"])
        selected = nodes[state["selected_node_id"]]
        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        semantic_hits, episodic_hits = lats.gather_node_memories(memory, state["question"], selected)
        return {
            "semantic_hits": semantic_hits,
            "episodic_hits": episodic_hits,
            "memory_counts": memory.counts(),
        }

    def expand_candidates(state: LATSLangGraphState) -> LATSLangGraphState:
        if not state.get("selected_node_id"):
            return {"expanded_child_ids": []}

        nodes = _deserialize_tree(state["tree_nodes"])
        selected = nodes[state["selected_node_id"]]
        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        expansion = lats.expand_node_with_llm(
            question=state["question"],
            llm=llm,
            memory=memory,
            tools=tools,
            selected_node=selected,
            branching_factor=state["branching_factor"],
            max_depth=state["max_depth"],
            semantic_hits=state.get("semantic_hits", []),
            episodic_hits=state.get("episodic_hits", []),
            verbose=True,
        )

        for child in expansion["children"]:
            nodes[child.node_id] = child

        return {
            "tree_nodes": _serialize_tree(nodes),
            "expanded_child_ids": [child.node_id for child in expansion["children"]],
            "total_tokens": state.get("total_tokens", 0) + expansion["tokens"],
            "llm_calls": state.get("llm_calls", 0) + expansion["llm_calls"],
            "tree_size": len(nodes),
        }

    def evaluate_frontier(state: LATSLangGraphState) -> LATSLangGraphState:
        nodes = _deserialize_tree(state["tree_nodes"])
        final_node: lats.SearchNode | None = None

        for child_id in state.get("expanded_child_ids", []):
            child = nodes[child_id]
            lats.backpropagate_score(nodes, child_id, child.heuristic_score)
            if child.final_answer and lats.score_search_node(child, state["question"]) >= 1.0:
                final_node = child

        best_node = final_node or lats.choose_best_node(nodes, state["question"])
        result: LATSLangGraphState = {
            "tree_nodes": _serialize_tree(nodes),
            "best_node_id": best_node.node_id,
            "iteration": state.get("iteration", 0) + 1,
            "tree_size": len(nodes),
        }
        if final_node is not None:
            result["final_answer"] = final_node.final_answer
            result["trajectory"] = final_node.trajectory
            result["steps"] = final_node.step
        print("")
        return result

    def finalize(state: LATSLangGraphState) -> LATSLangGraphState:
        memory = react_coala.CoALAMemoryStore(Path(state["memory_dir"]))
        nodes = _deserialize_tree(state["tree_nodes"])
        best_node = nodes[state["best_node_id"]] if state.get("best_node_id") else lats.choose_best_node(nodes, state["question"])

        final_answer = state.get("final_answer")
        extra_tokens = 0
        extra_llm_calls = 0
        if not final_answer:
            final_answer, extra_tokens, extra_llm_calls = lats.resolve_best_answer_from_node(
                question=state["question"],
                llm=llm,
                node=best_node,
                tools=tools,
                memory=memory,
            )

        # Persistencia CoALA: a melhor trajetoria encontrada vira episodio e resposta consolidada.
        memory.add_episode(
            state["question"],
            final_answer,
            react_coala._summarize_trajectory(best_node.trajectory),
        )
        react_coala._auto_consolidate_semantic_memory(memory, state["question"], final_answer)

        print("\n=== RESPOSTA FINAL ENCONTRADA ===")
        return {
            "final_answer": final_answer,
            "total_tokens": state.get("total_tokens", 0) + extra_tokens,
            "llm_calls": state.get("llm_calls", 0) + extra_llm_calls,
            "memory_counts": memory.counts(),
            "trajectory": best_node.trajectory,
            "steps": best_node.step,
            "tree_size": len(nodes),
        }

    def select_route(state: LATSLangGraphState) -> Literal["recall_memories", "finalize"]:
        if state.get("selected_node_id"):
            return "recall_memories"
        return "finalize"

    def evaluate_route(state: LATSLangGraphState) -> Literal["select_frontier", "finalize"]:
        if state.get("final_answer"):
            return "finalize"
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        return "select_frontier"

    # O grafo explicita o ciclo do LATS: selecionar folha -> recuperar memoria -> expandir -> avaliar -> repetir.
    graph_builder.add_node("bootstrap", bootstrap)
    graph_builder.add_node("select_frontier", select_frontier)
    graph_builder.add_node("recall_memories", recall_memories)
    graph_builder.add_node("expand_candidates", expand_candidates)
    graph_builder.add_node("evaluate_frontier", evaluate_frontier)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "bootstrap")
    graph_builder.add_edge("bootstrap", "select_frontier")
    graph_builder.add_conditional_edges(
        "select_frontier",
        select_route,
        {
            "recall_memories": "recall_memories",
            "finalize": "finalize",
        },
    )
    graph_builder.add_edge("recall_memories", "expand_candidates")
    graph_builder.add_edge("expand_candidates", "evaluate_frontier")
    # Depois de avaliar os novos ramos, ou seguimos expandindo a arvore ou encerramos com o melhor ramo.
    graph_builder.add_conditional_edges(
        "evaluate_frontier",
        evaluate_route,
        {
            "select_frontier": "select_frontier",
            "finalize": "finalize",
        },
    )
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=MemorySaver())


def invoke_lats_langgraph_once(
    config: PortkeyLangGraphConfig,
    question: str,
    max_iterations: int = 4,
    branching_factor: int = 2,
    max_depth: int = 4,
    exploration_weight: float = 1.2,
    memory_dir: str | Path = lats.DEFAULT_LATS_MEMORY_DIR,
    graph: Any | None = None,
) -> dict[str, Any]:
    graph = graph or build_lats_langgraph(config)
    started_at = time.perf_counter()
    result = graph.invoke(
        {
            "question": question,
            "max_iterations": max_iterations,
            "branching_factor": branching_factor,
            "max_depth": max_depth,
            "exploration_weight": exploration_weight,
            "memory_dir": str(memory_dir),
        },
        config={"configurable": {"thread_id": f"{config.thread_id}-lats-langgraph"}},
    )
    return {
        "trace_id": config.trace_id,
        "thread_id": f"{config.thread_id}-lats-langgraph",
        "provider": config.provider,
        "model": config.model,
        "resposta": result["final_answer"],
        "iterations": result["iteration"],
        "steps": result.get("steps", 0),
        "tokens": result["total_tokens"],
        "llm_calls": result.get("llm_calls", 0),
        "total_time_seconds": round(time.perf_counter() - started_at, 4),
        "trajectory": result.get("trajectory", []),
        "memory_dir": str(Path(memory_dir).resolve()),
        "memory_counts": result.get("memory_counts", {}),
        "tree_size": result.get("tree_size", 0),
    }


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv(
        "LATS_USER_MESSAGE",
        os.getenv("REACT_USER_MESSAGE", config.user_message or react_coala.OFFICIAL_BENCHMARK_QUESTION),
    )
    max_iterations = int(os.getenv("LATS_MAX_ITERATIONS", "4"))
    branching_factor = int(os.getenv("LATS_BRANCHING_FACTOR", "2"))
    max_depth = int(os.getenv("LATS_MAX_DEPTH", "4"))
    exploration_weight = float(os.getenv("LATS_EXPLORATION_WEIGHT", "1.2"))
    memory_dir = Path(os.getenv("LATS_MEMORY_DIR", lats.DEFAULT_LATS_MEMORY_DIR))
    graph_mermaid = Path(os.getenv("LATS_LANGGRAPH_MERMAID", DEFAULT_GRAPH_MERMAID))
    graph_png_value = os.getenv("LATS_LANGGRAPH_PNG", "").strip()

    graph = build_lats_langgraph(config)
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

    result = invoke_lats_langgraph_once(
        config=config,
        question=question,
        max_iterations=max_iterations,
        branching_factor=branching_factor,
        max_depth=max_depth,
        exploration_weight=exploration_weight,
        memory_dir=memory_dir,
        graph=graph,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{graph_message}")


if __name__ == "__main__":
    main()
