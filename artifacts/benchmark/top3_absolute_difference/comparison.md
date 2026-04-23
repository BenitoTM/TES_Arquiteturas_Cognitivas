# Benchmark comparativo

Pergunta usada: Pesquise os 3 países com maior PIB da América do Sul. Calcule a média do PIB per capita deles. Depois compare essa média com a média mundial do PIB per capita e responda qual é a diferença em US$. Por fim, diga se essa diferença é maior ou menor que 1000 US$.

## Referencia deterministica

- Top 3 por PIB: Brasil, Argentina, Venezuela
- Media do PIB per capita do top 3: 13475.32 US$
- Media mundial do PIB per capita (2024): 13631.20 US$
- Diferenca bruta: -155.88 US$
- Diferenca absoluta: 155.88 US$
- Comparacao com 1000 US$: menor

## Tabela comparativa

| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Sim | 6 | 14.3921 | 10377 | 732 | 11109 | 0.0049431 | 6 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Sim | 6 | 11.7678 | 9094 | 527 | 9621 | 0.0040457 | 6 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |
| LATS + CoALA | Nao | 11 | 22.5392 | 14700 | 885 | 15585 | 0.0066225 | 3 | working + semantic + episodic + procedural + arvore de busca | alta | quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora |

## Custo estimado

- Modelo considerado: `google/gemini-2.5-flash`
- O custo e estimado a partir de input/output tokens reportados pela API e nao representa necessariamente a fatura exata do Portkey/provedor.
- Fonte de preco usada: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing

## Notas de avaliacao

- ReAct + CoALA: A resposta contem os paises corretos do top 3, a media do top 3, a media mundial, a diferenca calculada e a comparacao correta com 1000 US$.
- Reflection / Reflexion: A resposta contem os paises corretos do top 3, a media do top 3, a media mundial, a diferenca calculada e a comparacao correta com 1000 US$.
- LATS + CoALA: faltaram paises do top 3: brasil, argentina, venezuela ; faltou a media calculada do top 3 ; faltou a media mundial ; faltou a diferenca calculada em US$ ; faltou citar o limite de 1000 US$ ; faltou indicar que a diferenca absoluta e menor que 1000 US$