import json
import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents.react_coala import OFFICIAL_BENCHMARK_QUESTION
from cognitive_lab.agents.reflection import DEFAULT_REFLECTION_MEMORY_DIR, run_reflection_agent


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
        llm = build_chat_model(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv(
        "REFLECTION_USER_MESSAGE",
        os.getenv("REACT_USER_MESSAGE", config.user_message or OFFICIAL_BENCHMARK_QUESTION),
    )
    max_attempts = int(os.getenv("REFLECTION_MAX_ATTEMPTS", "3"))
    max_steps = int(os.getenv("REFLECTION_MAX_STEPS", "6"))
    memory_dir = Path(os.getenv("REFLECTION_MEMORY_DIR", DEFAULT_REFLECTION_MEMORY_DIR))

    result = run_reflection_agent(
        question=question,
        llm=llm,
        max_attempts=max_attempts,
        max_steps=max_steps,
        memory_dir=memory_dir,
    )

    print("\n--- Resultado Final ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
