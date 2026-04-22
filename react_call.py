import json
import os
from pathlib import Path

from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.agents.react_coala import DEFAULT_MEMORY_DIR, run_react_coala_agent


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

    question = os.getenv("REACT_USER_MESSAGE", config.user_message or DEFAULT_QUESTION)
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
