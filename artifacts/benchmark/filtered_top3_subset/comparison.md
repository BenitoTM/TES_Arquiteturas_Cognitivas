# Benchmark comparativo

Pergunta usada: Pesquise os 3 países com maior PIB da América do Sul. Entre eles, considere apenas os países cujo PIB per capita é maior que a média mundial. Calcule a média do PIB per capita desse subconjunto e diga quantos países entraram nele. Depois responda se essa nova média é maior ou menor que a média mundial.

## Referencia deterministica

- Top 3 por PIB: Brasil, Argentina, Venezuela
- Paises no subconjunto: Argentina, Venezuela
- Quantidade no subconjunto: 2
- Media do subconjunto: 15065.55 US$
- Media mundial do PIB per capita (2024): 13631.20 US$
- Comparacao esperada: maior

## Tabela comparativa

| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Nao | 9 | 18.7850 | 12927 | 834 | 13761 | 0.0059631 | 9 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Nao | 36 | 116.5209 | 53994 | 3683 | 57677 | 0.0254057 | 8 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |
| LATS + CoALA | Nao | 11 | 43.2198 | 14159 | 1014 | 15173 | 0.0067827 | 3 | working + semantic + episodic + procedural + arvore de busca | alta | quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora |

## Custo estimado

- Modelo considerado: `google/gemini-2.5-flash`
- O custo e estimado a partir de input/output tokens reportados pela API e nao representa necessariamente a fatura exata do Portkey/provedor.
- Fonte de preco usada: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing

## Notas de avaliacao

- ReAct + CoALA: faltou dizer quantos paises entraram no subconjunto ; faltou a media calculada do subconjunto
- Reflection / Reflexion: faltaram paises do top 3: brasil, argentina, venezuela ; faltou dizer quantos paises entraram no subconjunto ; faltou a media calculada do subconjunto ; faltou a media mundial ; faltou indicar que a media do subconjunto e maior que a mundial
- LATS + CoALA: faltaram paises do top 3: brasil, argentina, venezuela ; faltou dizer quantos paises entraram no subconjunto ; faltou a media calculada do subconjunto ; faltou a media mundial ; faltou indicar que a media do subconjunto e maior que a mundial