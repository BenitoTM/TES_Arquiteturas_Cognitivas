# Suite comparativa de benchmarks

Modelo usado: `google/gemini-2.5-flash`

## Tabela consolidada

| Benchmark | Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Benchmark simples | ReAct + CoALA | Sim | 5 | 14.2772 | 7600 | 435 | 8035 | 0.0033675 |
| Benchmark simples | Reflection / Reflexion | Sim | 5 | 12.8387 | 7072 | 414 | 7486 | 0.0031566 |
| Benchmark simples | LATS + CoALA | Sim | 9 | 34.7414 | 11708 | 863 | 12571 | 0.0056699 |
| Benchmark com diferenca absoluta | ReAct + CoALA | Sim | 6 | 14.3921 | 10377 | 732 | 11109 | 0.0049431 |
| Benchmark com diferenca absoluta | Reflection / Reflexion | Sim | 6 | 11.7678 | 9094 | 527 | 9621 | 0.0040457 |
| Benchmark com diferenca absoluta | LATS + CoALA | Nao | 11 | 22.5392 | 14700 | 885 | 15585 | 0.0066225 |

## Artefatos individuais

- Benchmark simples: [comparison.md](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/simple_top3/comparison.md) | [comparison.json](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/simple_top3/comparison.json)
- Benchmark com diferenca absoluta: [comparison.md](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/top3_absolute_difference/comparison.md) | [comparison.json](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/artifacts/benchmark/top3_absolute_difference/comparison.json)