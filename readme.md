# TES arquiteturas cognitivas

Repositório de experimentos com arquiteturas cognitivas para agentes baseados em LLM, usando Portkey, LangChain e LangGraph.

## O que existe hoje

- um chatbot mínimo em LangGraph
- um agente ReACT com memória inspirada em CoALA
- o mesmo ReACT representado como um grafo LangGraph explícito
- um agente Reflection/Reflexion com múltiplas tentativas, julgamento e reflexão
- o mesmo Reflection representado como um grafo LangGraph explícito
- um agente LATS com busca em árvore, memória CoALA e seleção de ramos
- o mesmo LATS representado como um grafo LangGraph explícito

## Estrutura do repositório

```text
.
├── artifacts/
│   └── graphs/                  # Mermaid e PNG gerados pelos scripts LangGraph
├── cognitive_lab/
│   ├── agents/
│   │   ├── lats.py              # lógica do LATS + CoALA
│   │   ├── react_coala.py       # lógica do ReACT + CoALA
│   │   └── reflection.py        # lógica do Reflection/Reflexion
│   ├── runtime/
│   │   └── portkey.py           # integração com Portkey, modelo e chatbot base
│   ├── langgraph_portkey.py     # wrapper de compatibilidade
│   ├── lats_agent.py            # wrapper de compatibilidade
│   ├── react_coala.py           # wrapper de compatibilidade
│   └── reflection_agent.py      # wrapper de compatibilidade
├── data/
│   ├── coala_memory/            # memória persistente do ReACT
│   └── reflection_memory/       # memória persistente do Reflection
├── docs/
│   ├── analise_comparativa.md   # texto curto de comparação para o trabalho
│   ├── roteiro_apresentacao.md  # roteiro para a apresentação
│   └── free_llm_examples.md     # referências auxiliares
├── compare_agents.py            # benchmark comparativo automático
├── compare_agents_suite.py      # suíte com benchmark simples + benchmark filtrado
├── exemple.env                  # exemplo de configuração
├── llm_call.py                  # chatbot mínimo
├── lats_call.py                 # LATS + CoALA
├── lats_langgraph_call.py       # LATS + CoALA em LangGraph
├── react_call.py                # ReACT + CoALA
├── react_langgraph_call.py      # ReACT + CoALA em LangGraph
├── reflection_call.py           # Reflection/Reflexion
├── reflection_langgraph_call.py # Reflection em LangGraph
└── requirements.txt             # dependências pinadas
```

## Requisitos

- Python 3.13
- uma chave Portkey válida
- acesso à internet para o modelo e para a API do IBGE

## Instalação

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Copie o arquivo de exemplo e ajuste os valores:

```bash
cp exemple.env .env
```

## Configuração

Variáveis mínimas:

```bash
PORTKEY_API_KEY=...
PORTKEY_PROVIDER=@topicosdeengenhariadesoftwarre
PORTKEY_MODEL=z-ai/glm-4.5-air:free
PORTKEY_BASE_URL=https://api.portkey.ai/v1
```

Variáveis gerais opcionais:

```bash
SYSTEM_PROMPT=You are a helpful assistant.
USER_MESSAGE=Hello chatgpt, how are you?
LANGGRAPH_THREAD_ID=default
PORTKEY_TRACE_ID=custom-trace-id
PORTKEY_USER_ID=user-123
APP_ENV=development
MODEL_TEMPERATURE=0
```

## Como rodar

### Pergunta oficial usada no benchmark

```text
Pesquise os 3 países com maior PIB da América do Sul, calcule a média do PIB per capita deles e responda: essa média é maior ou menor que a média mundial?
```

### 1. Chatbot mínimo

```bash
.venv/bin/python llm_call.py
```

Saídas:

- Mermaid em `artifacts/graphs/chatbot_graph.mmd`
- PNG opcional com `CHATBOT_LANGGRAPH_PNG=artifacts/graphs/chatbot_graph.png`

### 2. ReACT + CoALA

```bash
.venv/bin/python react_call.py
```

Variáveis úteis:

```bash
REACT_USER_MESSAGE="Sua pergunta aqui"
REACT_MAX_STEPS=10
COALA_MEMORY_DIR=data/coala_memory
```

O agente usa:

- working memory para o estado atual da execução
- semantic memory em `data/coala_memory/semantic_memory.json`
- episodic memory em `data/coala_memory/episodic_memory.json`
- ferramentas `buscar_ibge`, `calcular`, `recordar_semantica`, `recordar_episodios`, `memorizar_semantica`

### 3. ReACT + CoALA em LangGraph

```bash
.venv/bin/python react_langgraph_call.py
```

Nós do grafo:

- `bootstrap`
- `recall_memories`
- `planner`
- `execute_action`
- `record_step`
- `finalize`

Saídas:

- Mermaid em `artifacts/graphs/react_langgraph_graph.mmd`
- PNG opcional com `REACT_LANGGRAPH_PNG=artifacts/graphs/react_langgraph_graph.png`

### 4. Reflection / Reflexion

```bash
.venv/bin/python reflection_call.py
```

Variáveis úteis:

```bash
REFLECTION_USER_MESSAGE="Sua pergunta aqui"
REFLECTION_MAX_ATTEMPTS=3
REFLECTION_MAX_STEPS=6
REFLECTION_MEMORY_DIR=data/reflection_memory
```

O agente executa:

- uma tentativa de solução
- um juiz que classifica a tentativa como `ACCEPT` ou `RETRY`
- uma reflexão curta e persistente quando precisa tentar de novo

Persistência:

- reflexões em `data/reflection_memory/reflections.json`
- memória semântica em `data/reflection_memory/semantic_memory.json`
- memória episódica em `data/reflection_memory/episodic_memory.json`

### 5. Reflection em LangGraph

```bash
.venv/bin/python reflection_langgraph_call.py
```

Nós do grafo:

- `bootstrap`
- `recall_reflections`
- `actor`
- `execute_action`
- `record_step`
- `judge`
- `reflect`
- `finalize`

Saídas:

- Mermaid em `artifacts/graphs/reflection_langgraph_graph.mmd`
- PNG opcional com `REFLECTION_LANGGRAPH_PNG=artifacts/graphs/reflection_langgraph_graph.png`

### 6. LATS + CoALA

```bash
.venv/bin/python lats_call.py
```

Variáveis úteis:

```bash
LATS_USER_MESSAGE="Sua pergunta aqui"
LATS_MAX_ITERATIONS=4
LATS_BRANCHING_FACTOR=2
LATS_MAX_DEPTH=4
LATS_MEMORY_DIR=data/lats_memory
```

O agente executa:

- seleção de um ramo promissor da árvore
- expansão com múltiplos candidatos por iteração
- avaliação heurística e retropropagação do score
- consolidação da melhor trajetória final com memória CoALA

Persistência:

- memória semântica em `data/lats_memory/semantic_memory.json`
- memória episódica em `data/lats_memory/episodic_memory.json`

### 7. LATS em LangGraph

```bash
.venv/bin/python lats_langgraph_call.py
```

Nós do grafo:

- `bootstrap`
- `select_frontier`
- `recall_memories`
- `expand_candidates`
- `evaluate_frontier`
- `finalize`

Saídas:

- Mermaid em `artifacts/graphs/lats_langgraph_graph.mmd`
- PNG opcional com `LATS_LANGGRAPH_PNG=artifacts/graphs/lats_langgraph_graph.png`

### 8. Benchmark comparativo

```bash
.venv/bin/python compare_agents.py
```

Esse script:

- roda `ReAct + CoALA`, `Reflection / Reflexion` e `LATS + CoALA` com a mesma pergunta oficial
- mede `resposta correta?`, `llm_calls`, `total_time_seconds`, `input_tokens`, `output_tokens`, `tokens`, `estimated_cost_usd` e `steps`
- salva artefatos em:
  - `artifacts/benchmark/comparison.md`
  - `artifacts/benchmark/comparison.json`

Esse é o comando principal para reproduzir a comparação final do trabalho.

### 9. Suíte com dois benchmarks

```bash
.venv/bin/python compare_agents_suite.py
```

Esse script roda automaticamente:

- o benchmark simples original
- o benchmark com filtro sobre o subconjunto dos 3 maiores PIBs

Artefatos gerados:

- `artifacts/benchmark/simple_top3/comparison.md`
- `artifacts/benchmark/simple_top3/comparison.json`
- `artifacts/benchmark/filtered_top3_subset/comparison.md`
- `artifacts/benchmark/filtered_top3_subset/comparison.json`
- `artifacts/benchmark/suite_summary.md`
- `artifacts/benchmark/suite_summary.json`

## Como salvar os grafos em PNG

Exemplos:

```bash
CHATBOT_LANGGRAPH_PNG=artifacts/graphs/chatbot_graph.png .venv/bin/python llm_call.py
LATS_LANGGRAPH_PNG=artifacts/graphs/lats_langgraph_graph.png .venv/bin/python lats_langgraph_call.py
REACT_LANGGRAPH_PNG=artifacts/graphs/react_langgraph_graph.png .venv/bin/python react_langgraph_call.py
REFLECTION_LANGGRAPH_PNG=artifacts/graphs/reflection_langgraph_graph.png .venv/bin/python reflection_langgraph_call.py
```

Se a renderização PNG falhar no ambiente, o Mermaid textual continua sendo salvo normalmente.

## Dependências

O `requirements.txt` está completo e pinado com as versões atualmente usadas no ambiente para facilitar reprodução.

Principais pacotes do projeto:

- `langchain`
- `langchain-openai`
- `langgraph`
- `portkey-ai`
- `python-dotenv`
- `requests`
- dependências transitivas pinadas como `openai`, `httpx`, `langchain-core`, `langgraph-checkpoint`, `pydantic` e outras

## Observações

- `cognitive_lab/langgraph_portkey.py`, `cognitive_lab/lats_agent.py`, `cognitive_lab/react_coala.py` e `cognitive_lab/reflection_agent.py` continuam existindo como wrappers de compatibilidade.
- `data/` e `artifacts/graphs/` estão no `.gitignore`, porque são saídas geradas em execução.
- `artifacts/benchmark/` também é gerado durante os benchmarks comparativos.
- `docs/free_llm_examples.md` foi mantido como material de referência complementar.
- A análise comparativa final está em [`docs/analise_comparativa.md`](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/docs/analise_comparativa.md:1).
- O roteiro de apresentação está em [`docs/roteiro_apresentacao.md`](/Users/benito/PycharmProjects/TES_arquiteturas_cognitivas/docs/roteiro_apresentacao.md:1).
