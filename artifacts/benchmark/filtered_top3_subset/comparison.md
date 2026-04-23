# Benchmark comparativo

Pergunta usada: Pesquise os 3 países com maior PIB da América do Sul.Calcule a média do PIB per capita deles.Depois compare essa média com a média mundial do PIB per capita e responda qual é a diferença em US$.Por fim, diga se essa diferença é maior ou menor que 1000 US$.

## Tipo de avaliacao

- Benchmark customizado avaliado por juiz LLM.
- Para prompts fora dos benchmarks determinísticos, a coluna de correção usa avaliação textual do modelo.

## Tabela comparativa

| Arquitetura | Resposta correta? | Chamadas ao LLM | Tempo total (s) | Input tokens | Output tokens | Tokens | Custo estimado (USD) | Steps | Memoria CoALA | Complexidade de codigo | Quando usar? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Nao | 7 | 12.5867 | 10166 | 764 | 10930 | 0.0049598 | 7 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Nao | 33 | 55.7539 | 48133 | 3622 | 51755 | 0.0234949 | 7 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |
| LATS + CoALA | Nao | 11 | 18.1619 | 13841 | 752 | 14593 | 0.0060323 | 3 | working + semantic + episodic + procedural + arvore de busca | alta | quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora |

## Custo estimado

- Modelo considerado: `google/gemini-2.5-flash`
- O custo e estimado a partir de input/output tokens reportados pela API e nao representa necessariamente a fatura exata do Portkey/provedor.
- Fonte de preco usada: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing

## Notas de avaliacao

- ReAct + CoALA: A resposta está incorreta em relação aos países com maior PIB da América do Sul e, consequentemente, nos cálculos subsequentes. A Venezuela não está entre os 3 maiores PIBs da América do Sul. Além disso, a resposta não apresenta os PIBs per capita individuais dos países para que o cálculo da média possa ser verificado.
- Reflection / Reflexion: A resposta do agente indica que não foi possível obter uma resposta, o que significa que nenhum dos pedidos da pergunta foi atendido.
- LATS + CoALA: A resposta do agente indica que não foi possível consolidar uma resposta, o que significa que não atendeu a nenhum dos pedidos da pergunta.