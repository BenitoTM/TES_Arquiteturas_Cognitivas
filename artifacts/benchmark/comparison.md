# Benchmark comparativo

Pergunta usada: Pesquise os 3 países com maior PIB da América do Sul, calcule a média do PIB per capita deles e responda: essa média é maior ou menor que a média mundial?

## Referencia deterministica

- Top 3 por PIB: Brasil, Argentina, Venezuela
- Media do PIB per capita do top 3: 13475.32 US$
- Media mundial do PIB per capita (2024): 13631.20 US$
- Comparacao esperada: menor

## Tabela comparativa

| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Tokens | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |
|---|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Sim | 5 | 9.4749 | 6727 | 5 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Sim | 6 | 11.5765 | 8403 | 6 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |

## Notas de avaliacao

- ReAct + CoALA: A resposta contem os tres paises corretos, a media do top 3, a media mundial e a comparacao correta.
- Reflection / Reflexion: A resposta contem os tres paises corretos, a media do top 3, a media mundial e a comparacao correta.