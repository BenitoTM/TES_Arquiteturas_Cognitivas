import json
import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents.react_coala import (
    DEFAULT_MEMORY_DIR,
    OFFICIAL_BENCHMARK_QUESTION,
    run_react_coala_agent,
)


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
        llm = build_chat_model(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv("REACT_USER_MESSAGE", config.user_message or OFFICIAL_BENCHMARK_QUESTION)
    max_steps = int(os.getenv("REACT_MAX_STEPS", "10"))
    memory_dir = Path(os.getenv("COALA_MEMORY_DIR", DEFAULT_MEMORY_DIR))

    result = run_react_coala_agent(
        question=question,
        llm=llm,
        max_steps=max_steps,
        memory_dir=memory_dir,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
