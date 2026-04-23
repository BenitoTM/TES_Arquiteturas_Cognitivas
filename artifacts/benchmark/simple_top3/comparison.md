# Benchmark comparativo

Pergunta usada: Pesquise os 3 países com maior PIB da América do Sul, calcule a média do PIB per capita deles e responda: essa média é maior ou menor que a média mundial?

## Referencia deterministica

- Top 3 por PIB: Brasil, Argentina, Venezuela
- Media do PIB per capita do top 3: 13475.32 US$
- Media mundial do PIB per capita (2024): 13631.20 US$
- Comparacao esperada: menor

## Tabela comparativa

| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Sim | 5 | 18.8484 | 6294 | 433 | 6727 | 0.0029707 | 5 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Sim | 6 | 14.4895 | 7907 | 496 | 8403 | 0.0036121 | 6 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |
| LATS + CoALA | Sim | 8 | 35.8405 | 9492 | 554 | 10046 | 0.0042326 | 4 | working + semantic + episodic + procedural + arvore de busca | alta | quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora |

## Custo estimado

- Modelo considerado: `google/gemini-2.5-flash`
- O custo e estimado a partir de input/output tokens reportados pela API e nao representa necessariamente a fatura exata do Portkey/provedor.
- Fonte de preco usada: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing

## Notas de avaliacao

- ReAct + CoALA: A resposta contem os tres paises corretos, a media do top 3, a media mundial e a comparacao correta.
- Reflection / Reflexion: A resposta contem os tres paises corretos, a media do top 3, a media mundial e a comparacao correta.
- LATS + CoALA: A resposta contem os tres paises corretos, a media do top 3, a media mundial e a comparacao correta.