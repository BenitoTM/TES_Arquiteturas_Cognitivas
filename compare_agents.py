from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cognitive_lab.agents import react_coala, reflection
from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model


DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark")
REACT_BENCHMARK_MEMORY_DIR = Path("data/benchmark/react_coala")
REFLECTION_BENCHMARK_MEMORY_DIR = Path("data/benchmark/reflection")


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def _is_response_correct(response: str, reference: dict[str, Any]) -> tuple[bool, str]:
    evaluation = react_coala.evaluate_official_benchmark_answer(response, reference=reference)
    return evaluation["correct"], evaluation["feedback"]


def _benchmark_metadata() -> dict[str, dict[str, str]]:
    return {
        "ReAct + CoALA": {
            "memory_type": "working + semantic + episodic + procedural",
            "complexity": "media",
            "when_to_use": "quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation",
        },
        "Reflection / Reflexion": {
            "memory_type": "working + semantic + episodic + procedural + reflection memory",
            "complexity": "alta",
            "when_to_use": "quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas",
        },
    }


def _to_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Tokens | Steps | "
        "Memoria CoALA | Complexidade de codigo | Quando usar? |\n"
        "|---|---:|---:|---:|---:|---:|---|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row['architecture']} | "
            f"{'Sim' if row['correct'] else 'Nao'} | "
            f"{row['llm_calls']} | "
            f"{row['total_time_seconds']:.4f} | "
            f"{row['tokens']} | "
            f"{row['steps']} | "
            f"{row['memory_type']} | "
            f"{row['complexity']} | "
            f"{row['when_to_use']} |"
        )
    return "\n".join(lines)


def main() -> None:
    config = PortkeyLangGraphConfig.from_env()
    llm = build_chat_model(config)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_question = react_coala.OFFICIAL_BENCHMARK_QUESTION
    reference = react_coala.get_official_benchmark_reference()

    _reset_directory(REACT_BENCHMARK_MEMORY_DIR)
    react_result = react_coala.run_react_coala_agent(
        question=benchmark_question,
        llm=llm,
        max_steps=10,
        memory_dir=REACT_BENCHMARK_MEMORY_DIR,
    )

    _reset_directory(REFLECTION_BENCHMARK_MEMORY_DIR)
    reflection_result = reflection.run_reflection_agent(
        question=benchmark_question,
        llm=llm,
        max_attempts=3,
        max_steps=6,
        memory_dir=REFLECTION_BENCHMARK_MEMORY_DIR,
    )

    metadata = _benchmark_metadata()
    rows = []
    raw_results = {
        "question": benchmark_question,
        "reference": reference,
        "results": {
            "react": react_result,
            "reflection": reflection_result,
        },
    }

    for architecture, result in (
        ("ReAct + CoALA", react_result),
        ("Reflection / Reflexion", reflection_result),
    ):
        correct, evaluation_note = _is_response_correct(result["resposta"], reference)
        rows.append(
            {
                "architecture": architecture,
                "correct": correct,
                "evaluation_note": evaluation_note,
                "llm_calls": result["llm_calls"],
                "total_time_seconds": float(result["total_time_seconds"]),
                "tokens": result["tokens"],
                "steps": result["steps"],
                "memory_type": metadata[architecture]["memory_type"],
                "complexity": metadata[architecture]["complexity"],
                "when_to_use": metadata[architecture]["when_to_use"],
                "answer": result["resposta"],
            }
        )

    markdown_report = "\n".join(
        [
            "# Benchmark comparativo",
            "",
            f"Pergunta usada: {benchmark_question}",
            "",
            "## Referencia deterministica",
            "",
            f"- Top 3 por PIB: {', '.join(item['country_name'] for item in reference['top3'])}",
            f"- Media do PIB per capita do top 3: {reference['top3_average']:.2f} US$",
            f"- Media mundial do PIB per capita ({reference['world_year']}): {reference['world_average']:.2f} US$",
            f"- Comparacao esperada: {reference['comparison']}",
            "",
            "## Tabela comparativa",
            "",
            _to_markdown_table(rows),
            "",
            "## Notas de avaliacao",
            "",
            *[f"- {row['architecture']}: {row['evaluation_note']}" for row in rows],
        ]
    )

    (output_dir / "comparison.json").write_text(
        json.dumps({"rows": rows, **raw_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(markdown_report, encoding="utf-8")

    print(markdown_report)
    print(f"\nJSON salvo em: {(output_dir / 'comparison.json').resolve()}")
    print(f"Markdown salvo em: {(output_dir / 'comparison.md').resolve()}")


if __name__ == "__main__":
    main()
