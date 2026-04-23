import json
import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents.lats import DEFAULT_LATS_MEMORY_DIR, run_lats_agent
from cognitive_lab.agents.react_coala import OFFICIAL_BENCHMARK_QUESTION


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
        llm = build_chat_model(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv(
        "LATS_USER_MESSAGE",
        os.getenv("REACT_USER_MESSAGE", config.user_message or OFFICIAL_BENCHMARK_QUESTION),
    )
    max_iterations = int(os.getenv("LATS_MAX_ITERATIONS", "4"))
    branching_factor = int(os.getenv("LATS_BRANCHING_FACTOR", "2"))
    max_depth = int(os.getenv("LATS_MAX_DEPTH", "4"))
    memory_dir = Path(os.getenv("LATS_MEMORY_DIR", DEFAULT_LATS_MEMORY_DIR))

    result = run_lats_agent(
        question=question,
        llm=llm,
        max_iterations=max_iterations,
        branching_factor=branching_factor,
        max_depth=max_depth,
        memory_dir=memory_dir,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
