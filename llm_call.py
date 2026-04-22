import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, invoke_graph_once, build_chat_graph


DEFAULT_CHATBOT_GRAPH_MERMAID = "artifacts/graphs/chatbot_graph.mmd"


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
        result = invoke_graph_once(config)
        graph = build_chat_graph(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(result["response"])
    print(result)
    graph_mermaid = Path(os.getenv("CHATBOT_LANGGRAPH_MERMAID", DEFAULT_CHATBOT_GRAPH_MERMAID))
    graph_png_value = os.getenv("CHATBOT_LANGGRAPH_PNG", "").strip()

    graph_mermaid.parent.mkdir(parents=True, exist_ok=True)
    graph_mermaid.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Mermaid salvo em: {graph_mermaid.resolve()}")

    if graph_png_value:
        graph_png = Path(graph_png_value)
        try:
            graph_png.parent.mkdir(parents=True, exist_ok=True)
            png_bytes = graph.get_graph().draw_mermaid_png()
            graph_png.write_bytes(png_bytes)
            print(f"PNG salvo em: {graph_png.resolve()}")
        except Exception as exc:
            print(f"Nao foi possivel gerar o PNG do grafo: {exc}")



if __name__ == "__main__":
    main()
