from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from cognitive_lab.agents import lats, react_coala, reflection
from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model
from cognitive_lab.runtime.pricing import estimate_cost_usd


DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark")
REACT_BENCHMARK_MEMORY_DIR = Path("data/benchmark/react_coala")
REFLECTION_BENCHMARK_MEMORY_DIR = Path("data/benchmark/reflection")
LATS_BENCHMARK_MEMORY_DIR = Path("data/benchmark/lats")


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _parse_judge_verdict(text: str) -> tuple[bool, str]:
    verdict_match = re.search(r"Verdict:\s*(ACCEPT|RETRY)", text)
    feedback_match = re.search(r"Feedback:\s*(.+)", text, re.DOTALL)
    verdict = verdict_match.group(1).strip() if verdict_match else "RETRY"
    feedback = feedback_match.group(1).strip() if feedback_match else "Sem feedback detalhado."
    return verdict == "ACCEPT", feedback


def _judge_custom_response(question: str, response: str, llm: Any) -> tuple[bool, str]:
    judged = llm.invoke(
        [
            SystemMessage(content="Voce avalia respostas de benchmark com rigor e objetividade."),
            HumanMessage(
                content=f"""Pergunta:
{question}

Resposta do agente:
{response}

Avalie se a resposta:
- responde todos os pedidos da pergunta;
- inclui os calculos ou conclusoes necessarias;
- e clara e coerente com os numeros citados;
- nao deixou de fora condicoes, filtros ou comparacoes importantes.

Responda exatamente neste formato:
Verdict: ACCEPT ou RETRY
Feedback: [explique objetivamente o que faltou ou por que a resposta foi aceita]
"""
            ),
        ]
    )
    text = judged.content if isinstance(judged.content, str) else str(judged.content)
    return _parse_judge_verdict(text)


def _is_response_correct(
    question: str,
    response: str,
    reference: dict[str, Any] | None,
    llm: Any,
) -> tuple[bool, str]:
    deterministic = react_coala.evaluate_benchmark_answer(question, response, reference=reference)
    if deterministic is not None:
        return deterministic["correct"], deterministic["feedback"]
    return _judge_custom_response(question, response, llm)


def _reference_section(question: str, reference: dict[str, Any] | None) -> list[str]:
    normalized = react_coala._normalize_text(question)

    if normalized == react_coala._normalize_text(react_coala.OFFICIAL_BENCHMARK_QUESTION) and reference is not None:
        return [
            "## Referencia deterministica",
            "",
            f"- Top 3 por PIB: {', '.join(item['country_name'] for item in reference['top3'])}",
            f"- Media do PIB per capita do top 3: {reference['top3_average']:.2f} US$",
            f"- Media mundial do PIB per capita ({reference['world_year']}): {reference['world_average']:.2f} US$",
            f"- Comparacao esperada: {reference['comparison']}",
            "",
        ]

    if react_coala._normalize_text(question) == react_coala._normalize_text(react_coala.FILTERED_TOP3_BENCHMARK_QUESTION) and reference is not None:
        return [
            "## Referencia deterministica",
            "",
            f"- Top 3 por PIB: {', '.join(item['country_name'] for item in reference['top3'])}",
            f"- Paises no subconjunto: {', '.join(item['country_name'] for item in reference['eligible_subset'])}",
            f"- Quantidade no subconjunto: {reference['eligible_count']}",
            f"- Media do subconjunto: {reference['subset_average']:.2f} US$",
            f"- Media mundial do PIB per capita ({reference['world_year']}): {reference['world_average']:.2f} US$",
            f"- Comparacao esperada: {reference['comparison']}",
            "",
        ]

    if react_coala._normalize_text(question) == react_coala._normalize_text(react_coala.ABSOLUTE_DIFFERENCE_TOP3_BENCHMARK_QUESTION) and reference is not None:
        return [
            "## Referencia deterministica",
            "",
            f"- Top 3 por PIB: {', '.join(item['country_name'] for item in reference['top3'])}",
            f"- Media do PIB per capita do top 3: {reference['top3_average']:.2f} US$",
            f"- Media mundial do PIB per capita ({reference['world_year']}): {reference['world_average']:.2f} US$",
            f"- Diferenca bruta: {reference['raw_difference']:.2f} US$",
            f"- Diferenca absoluta: {reference['absolute_difference']:.2f} US$",
            f"- Comparacao com 1000 US$: {reference['threshold_comparison']}",
            "",
        ]

    return [
        "## Tipo de avaliacao",
        "",
        "- Benchmark customizado avaliado por juiz LLM.",
        "- Para prompts fora dos benchmarks determinísticos, a coluna de correção usa avaliação textual do modelo.",
        "",
    ]


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
        "LATS + CoALA": {
            "memory_type": "working + semantic + episodic + procedural + arvore de busca",
            "complexity": "alta",
            "when_to_use": "quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora",
        },
    }


def _to_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | "
        "Custo estimado (USD) | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row['architecture']} | "
            f"{'Sim' if row['correct'] else 'Nao'} | "
            f"{row['llm_calls']} | "
            f"{row['total_time_seconds']:.4f} | "
            f"{row['input_tokens']} | "
            f"{row['output_tokens']} | "
            f"{row['tokens']} | "
            f"{row['estimated_cost_usd'] if row['estimated_cost_usd'] is not None else 'N/D'} | "
            f"{row['steps']} | "
            f"{row['memory_type']} | "
            f"{row['complexity']} | "
            f"{row['when_to_use']} |"
        )
    return "\n".join(lines)


def load_settings_from_env() -> dict[str, Any]:
    return {
        "question": os.getenv("COMPARE_BENCHMARK_QUESTION", react_coala.OFFICIAL_BENCHMARK_QUESTION),
        "output_dir": Path(os.getenv("COMPARE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        "react_max_steps": int(os.getenv("COMPARE_REACT_MAX_STEPS", "10")),
        "reflection_max_attempts": int(os.getenv("COMPARE_REFLECTION_MAX_ATTEMPTS", "3")),
        "reflection_max_steps": int(os.getenv("COMPARE_REFLECTION_MAX_STEPS", "6")),
        "lats_max_iterations": int(os.getenv("COMPARE_LATS_MAX_ITERATIONS", "4")),
        "lats_branching_factor": int(os.getenv("COMPARE_LATS_BRANCHING_FACTOR", "2")),
        "lats_max_depth": int(os.getenv("COMPARE_LATS_MAX_DEPTH", "4")),
    }


def run_benchmark(
    *,
    config: PortkeyLangGraphConfig,
    llm: Any,
    question: str,
    output_dir: str | Path,
    react_max_steps: int,
    reflection_max_attempts: int,
    reflection_max_steps: int,
    lats_max_iterations: int,
    lats_branching_factor: int,
    lats_max_depth: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = react_coala.get_benchmark_reference(question)

    _reset_directory(REACT_BENCHMARK_MEMORY_DIR)
    react_result = react_coala.run_react_coala_agent(
        question=question,
        llm=llm,
        max_steps=react_max_steps,
        memory_dir=REACT_BENCHMARK_MEMORY_DIR,
    )

    _reset_directory(REFLECTION_BENCHMARK_MEMORY_DIR)
    reflection_result = reflection.run_reflection_agent(
        question=question,
        llm=llm,
        max_attempts=reflection_max_attempts,
        max_steps=reflection_max_steps,
        memory_dir=REFLECTION_BENCHMARK_MEMORY_DIR,
    )

    _reset_directory(LATS_BENCHMARK_MEMORY_DIR)
    lats_result = lats.run_lats_agent(
        question=question,
        llm=llm,
        max_iterations=lats_max_iterations,
        branching_factor=lats_branching_factor,
        max_depth=lats_max_depth,
        memory_dir=LATS_BENCHMARK_MEMORY_DIR,
    )

    metadata = _benchmark_metadata()
    rows = []
    raw_results = {
        "question": question,
        "reference": reference,
        "results": {
            "react": react_result,
            "reflection": reflection_result,
            "lats": lats_result,
        },
    }

    for architecture, result in (
        ("ReAct + CoALA", react_result),
        ("Reflection / Reflexion", reflection_result),
        ("LATS + CoALA", lats_result),
    ):
        correct, evaluation_note = _is_response_correct(question, result["resposta"], reference, llm)
        token_usage = result.get("token_usage", {})
        cost_estimate = estimate_cost_usd(config.model, token_usage)
        rows.append(
            {
                "architecture": architecture,
                "correct": correct,
                "evaluation_note": evaluation_note,
                "llm_calls": result["llm_calls"],
                "total_time_seconds": float(result["total_time_seconds"]),
                "input_tokens": int(token_usage.get("input_tokens", 0)),
                "output_tokens": int(token_usage.get("output_tokens", 0)),
                "cached_input_tokens": int(token_usage.get("cached_input_tokens", 0)),
                "tokens": result["tokens"],
                "estimated_cost_usd": cost_estimate["estimated_cost_usd"],
                "pricing_source": cost_estimate["pricing_source"],
                "pricing_reference": cost_estimate.get("pricing_reference"),
                "input_price_per_1m": cost_estimate["input_price_per_1m"],
                "output_price_per_1m": cost_estimate["output_price_per_1m"],
                "cached_input_price_per_1m": cost_estimate["cached_input_price_per_1m"],
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
            f"Pergunta usada: {question}",
            "",
            *_reference_section(question, reference),
            "## Tabela comparativa",
            "",
            _to_markdown_table(rows),
            "",
            "## Custo estimado",
            "",
            f"- Modelo considerado: `{config.model}`",
            "- O custo e estimado a partir de input/output tokens reportados pela API e nao representa necessariamente a fatura exata do Portkey/provedor.",
            f"- Fonte de preco usada: {next((row['pricing_reference'] for row in rows if row.get('pricing_reference')), 'N/D')}",
            "",
            "## Notas de avaliacao",
            "",
            *[f"- {row['architecture']}: {row['evaluation_note']}" for row in rows],
        ]
    )

    comparison_json_path = output_dir / "comparison.json"
    comparison_md_path = output_dir / "comparison.md"
    comparison_json_path.write_text(
        json.dumps({"rows": rows, **raw_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_md_path.write_text(markdown_report, encoding="utf-8")

    return {
        "question": question,
        "reference": reference,
        "rows": rows,
        "results": raw_results["results"],
        "comparison_md_path": str(comparison_md_path.resolve()),
        "comparison_json_path": str(comparison_json_path.resolve()),
        "markdown_report": markdown_report,
    }


def main() -> None:
    config = PortkeyLangGraphConfig.from_env()
    llm = build_chat_model(config)
    settings = load_settings_from_env()
    result = run_benchmark(
        config=config,
        llm=llm,
        question=settings["question"],
        output_dir=settings["output_dir"],
        react_max_steps=settings["react_max_steps"],
        reflection_max_attempts=settings["reflection_max_attempts"],
        reflection_max_steps=settings["reflection_max_steps"],
        lats_max_iterations=settings["lats_max_iterations"],
        lats_branching_factor=settings["lats_branching_factor"],
        lats_max_depth=settings["lats_max_depth"],
    )

    print(result["markdown_report"])
    print(f"\nJSON salvo em: {result['comparison_json_path']}")
    print(f"Markdown salvo em: {result['comparison_md_path']}")


if __name__ == "__main__":
    main()
