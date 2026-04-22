import json
import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents.reflection import DEFAULT_REFLECTION_MEMORY_DIR, run_reflection_agent


DEFAULT_QUESTION = (
    "Pesquise os 3 países com maior PIB da América do Sul, calcule a média do "
    "PIB per capita deles e responda: essa média é maior ou menor que a média mundial?"
)


def main() -> None:
    try:
        config = PortkeyLangGraphConfig.from_env()
        llm = build_chat_model(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    question = os.getenv("REFLECTION_USER_MESSAGE", os.getenv("REACT_USER_MESSAGE", config.user_message or DEFAULT_QUESTION))
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
