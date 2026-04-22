# TES arquiteturas cognitivas

Repositório de experimentos com arquiteturas cognitivas para agentes baseados em LLM, usando Portkey, LangChain e LangGraph.

## O que existe hoje

- um chatbot mínimo em LangGraph
- um agente ReACT com memória inspirada em CoALA
- o mesmo ReACT representado como um grafo LangGraph explícito
- um agente Reflection/Reflexion com múltiplas tentativas, julgamento e reflexão
- o mesmo Reflection representado como um grafo LangGraph explícito

## Estrutura do repositório

```text
.
├── artifacts/
│   └── graphs/                  # Mermaid e PNG gerados pelos scripts LangGraph
├── cognitive_lab/
│   ├── agents/
│   │   ├── react_coala.py       # lógica do ReACT + CoALA
│   │   └── reflection.py        # lógica do Reflection/Reflexion
│   ├── runtime/
│   │   └── portkey.py           # integração com Portkey, modelo e chatbot base
│   ├── langgraph_portkey.py     # wrapper de compatibilidade
│   ├── react_coala.py           # wrapper de compatibilidade
│   └── reflection_agent.py      # wrapper de compatibilidade
├── data/
│   ├── coala_memory/            # memória persistente do ReACT
│   └── reflection_memory/       # memória persistente do Reflection
├── docs/
│   └── free_llm_examples.md     # referências auxiliares
├── exemple.env                  # exemplo de configuração
├── llm_call.py                  # chatbot mínimo
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

## Como salvar os grafos em PNG

Exemplos:

```bash
CHATBOT_LANGGRAPH_PNG=artifacts/graphs/chatbot_graph.png .venv/bin/python llm_call.py
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

- `cognitive_lab/langgraph_portkey.py`, `cognitive_lab/react_coala.py` e `cognitive_lab/reflection_agent.py` continuam existindo como wrappers de compatibilidade.
- `data/` e `artifacts/graphs/` estão no `.gitignore`, porque são saídas geradas em execução.
- `docs/free_llm_examples.md` foi mantido como material de referência complementar.
