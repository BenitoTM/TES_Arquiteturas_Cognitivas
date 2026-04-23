from __future__ import annotations

import json
from pathlib import Path

from compare_agents import run_benchmark
from cognitive_lab.agents import react_coala
from cognitive_lab.runtime.portkey import PortkeyLangGraphConfig, build_chat_model


DEFAULT_SUITE_OUTPUT_DIR = Path("artifacts/benchmark")


BENCHMARK_SPECS = [
    {
        "id": "simple_top3",
        "title": "Benchmark simples",
        "question": react_coala.OFFICIAL_BENCHMARK_QUESTION,
        "react_max_steps": 10,
        "reflection_max_attempts": 3,
        "reflection_max_steps": 6,
        "lats_max_iterations": 4,
        "lats_branching_factor": 2,
        "lats_max_depth": 4,
    },
    {
        "id": "filtered_top3_subset",
        "title": "Benchmark com filtro e subconjunto",
        "question": react_coala.FILTERED_TOP3_BENCHMARK_QUESTION,
        "react_max_steps": 12,
        "reflection_max_attempts": 4,
        "reflection_max_steps": 8,
        "lats_max_iterations": 5,
        "lats_branching_factor": 2,
        "lats_max_depth": 5,
    },
]


def _suite_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Benchmark | Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | "
        "Input tokens | Output tokens | Tokens | Custo estimado (USD) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row['benchmark']} | "
            f"{row['architecture']} | "
            f"{'Sim' if row['correct'] else 'Nao'} | "
            f"{row['llm_calls']} | "
            f"{row['total_time_seconds']:.4f} | "
            f"{row['input_tokens']} | "
            f"{row['output_tokens']} | "
            f"{row['tokens']} | "
            f"{row['estimated_cost_usd'] if row['estimated_cost_usd'] is not None else 'N/D'} |"
        )
    return "\n".join(lines)


def main() -> None:
    config = PortkeyLangGraphConfig.from_env()
    llm = build_chat_model(config)
    suite_output_dir = DEFAULT_SUITE_OUTPUT_DIR
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows: list[dict[str, object]] = []
    suite_results: list[dict[str, object]] = []

    for spec in BENCHMARK_SPECS:
        benchmark_output_dir = suite_output_dir / spec["id"]
        try:
            result = run_benchmark(
                config=config,
                llm=llm,
                question=spec["question"],
                output_dir=benchmark_output_dir,
                react_max_steps=spec["react_max_steps"],
                reflection_max_attempts=spec["reflection_max_attempts"],
                reflection_max_steps=spec["reflection_max_steps"],
                lats_max_iterations=spec["lats_max_iterations"],
                lats_branching_factor=spec["lats_branching_factor"],
                lats_max_depth=spec["lats_max_depth"],
            )

            suite_results.append(
                {
                    "id": spec["id"],
                    "title": spec["title"],
                    "question": spec["question"],
                    "status": "success",
                    "comparison_md_path": result["comparison_md_path"],
                    "comparison_json_path": result["comparison_json_path"],
                    "rows": result["rows"],
                }
            )
            for row in result["rows"]:
                suite_rows.append(
                    {
                        "benchmark": spec["title"],
                        **row,
                    }
                )
        except Exception as exc:
            benchmark_output_dir.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "id": spec["id"],
                "title": spec["title"],
                "question": spec["question"],
                "status": "error",
                "error": str(exc),
            }
            (benchmark_output_dir / "error.json").write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            suite_results.append(error_payload)

    suite_markdown = "\n".join(
        [
            "# Suite comparativa de benchmarks",
            "",
            f"Modelo usado: `{config.model}`",
            "",
            "## Tabela consolidada",
            "",
            _suite_table(suite_rows) if suite_rows else "Nenhum benchmark foi concluido com sucesso.",
            "",
            "## Artefatos individuais",
            "",
            *[
                (
                    f"- {item['title']}: [comparison.md]({item['comparison_md_path']}) | "
                    f"[comparison.json]({item['comparison_json_path']})"
                    if item.get("status") == "success"
                    else f"- {item['title']}: erro durante a execucao. Ver [error.json]({(suite_output_dir / item['id'] / 'error.json').resolve()})"
                )
                for item in suite_results
            ],
        ]
    )

    suite_json_path = suite_output_dir / "suite_summary.json"
    suite_md_path = suite_output_dir / "suite_summary.md"
    suite_json_path.write_text(
        json.dumps(
            {
                "model": config.model,
                "benchmarks": suite_results,
                "rows": suite_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    suite_md_path.write_text(suite_markdown, encoding="utf-8")

    print(suite_markdown)
    print(f"\nResumo JSON salvo em: {suite_json_path.resolve()}")
    print(f"Resumo Markdown salvo em: {suite_md_path.resolve()}")


if __name__ == "__main__":
    main()
