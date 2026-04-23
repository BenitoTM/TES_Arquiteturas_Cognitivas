# Suite comparativa de benchmarks

Modelo usado: `google/gemini-2.5-flash`

## Tabela consolidada

| Benchmark | Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Benchmark simples | ReAct + CoALA | Sim | 5 | 18.8484 | 6294 | 433 | 6727 | 0.0029707 |
| Benchmark simples | Reflection / Reflexion | Sim | 6 | 14.4895 | 7907 | 496 | 8403 | 0.0036121 |
| Benchmark simples | LATS + CoALA | Sim | 8 | 35.8405 | 9492 | 554 | 10046 | 0.0042326 |
| Benchmark com filtro e subconjunto | ReAct + CoALA | Nao | 9 | 18.7850 | 12927 | 834 | 13761 | 0.0059631 |
| Benchmark com filtro e subconjunto | Reflection / Reflexion | Nao | 36 | 116.5209 | 53994 | 3683 | 57677 | 0.0254057 |
| Benchmark com filtro e subconjunto | LATS + CoALA | Nao | 11 | 43.2198 | 14159 | 1014 | 15173 | 0.0067827 |

## Artefatos individuais

- Benchmark simples: [comparison.md](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/simple_top3/comparison.md) | [comparison.json](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/simple_top3/comparison.json)
- Benchmark com filtro e subconjunto: [comparison.md](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/filtered_top3_subset/comparison.md) | [comparison.json](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/filtered_top3_subset/comparison.json)