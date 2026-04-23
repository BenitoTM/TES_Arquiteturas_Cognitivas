# Outline Da Apresentacao

Este arquivo resume a narrativa ideal da apresentacao para a Aula 5. Ele foi montado a partir do repositório do projeto e do PDF `/Users/benito/Downloads/aula4_5_arquiteturas_cognitivas (2).pdf`. A apresentacao deve caber em `7-8 minutos` e incluir `pelo menos 1 demo ao vivo`.

## Tese Central

Implementamos `3 arquiteturas agenticas` sobre o mesmo problema de benchmark oficial do professor:

- `ReAct + CoALA`
- `Reflection / Reflexion`
- `LATS + CoALA`

O `CoALA` nao aparece como um quarto agente independente no projeto; ele foi usado como `lente conceitual` e como `organizacao de memoria`. No benchmark usado, as tres arquiteturas acertaram a resposta, mas com trade-offs claros:

- `ReAct` foi a opcao mais simples, rapida e barata
- `Reflection` adicionou revisao iterativa e maior robustez conceitual
- `LATS` tornou explicita a exploracao de caminhos alternativos, com maior custo

## Restricoes Que A Apresentacao Deve Respeitar

- Nao afirmar que o projeto implementou `4 agentes completos`; o correto e dizer que implementou `3 arquiteturas agenticas` e usou `CoALA` como framework conceitual.
- Usar como fonte principal dos resultados:
  - `artifacts/benchmark/comparison.json`
  - `artifacts/benchmark/comparison.md`
- Usar o PDF do professor como fonte principal para:
  - objetivo da atividade
  - estrutura da apresentacao
  - benchmark oficial
- Nao inventar numeros, anos, custos ou resultados fora do que esta nos artefatos.
- Se precisar escolher apenas uma demo ao vivo, priorizar `ReAct`, pois e a execucao mais curta e facil de explicar.

## Estrutura Recomendada Do Deck

### Slide 1 — Titulo E Objetivo

Abrir com o tema da atividade, os integrantes e uma frase-curta dizendo que o grupo comparou `ReAct`, `Reflection` e `LATS` no mesmo benchmark, usando `CoALA` como lente de analise.

Tempo sugerido: `40-50s`

### Slide 2 — O Que O Professor Pediu

Explicar em linguagem simples o enunciado da atividade:

- estudar arquiteturas cognitivas
- implementar pelo menos `ReAct + 1`
- comparar no mesmo problema
- apresentar resultados empiricos
- usar `CoALA` para classificar memoria, acoes e decisao

Tempo sugerido: `45s`

### Slide 3 — Arquiteturas Estudadas E Papel Do CoALA

Mostrar rapidamente:

- `ReAct`: Thought -> Action -> Observation
- `Reflection / Reflexion`: tentativa -> julgamento -> reflexao -> nova tentativa
- `LATS`: busca em arvore com avaliacao de ramos
- `CoALA`: framework para organizar `working`, `episodic`, `semantic` e `procedural memory`

Tempo sugerido: `60s`

### Slide 4 — Como O Projeto Foi Implementado

Apresentar o recorte tecnico do repositório:

- Python + Portkey + LangChain + LangGraph
- ferramentas externas para buscar dados e calcular
- memorias persistidas em JSON
- versoes em grafo para visualizacao

Tempo sugerido: `50s`

### Slide 5 — Benchmark E Demo

Mostrar a pergunta oficial do benchmark e explicar por que ela e boa para comparar agentes:

- exige busca
- exige calculo
- exige julgamento final

Neste slide, preparar a transicao para a demo. A recomendacao e executar `react_call.py` ao vivo e usar os artefatos ja salvos para os outros agentes.

Referencia de resposta do benchmark para citar no deck, se necessario:

- top 3 por PIB: `Brasil`, `Argentina`, `Venezuela`
- media do PIB per capita do top 3: `13475.32 US$`
- media mundial do PIB per capita: `13631.20 US$`
- comparacao final: `menor`

Tempo sugerido: `90-120s`

### Slide 6 — Resultados Comparativos

Mostrar a tabela principal com os dados reais do arquivo `artifacts/benchmark/comparison.md`. Mensagem central:

- as tres acertaram
- `ReAct` foi o mais eficiente
- `Reflection` custou mais para revisar
- `LATS` foi o mais caro, mas o mais deliberativo

Tempo sugerido: `75s`

### Slide 7 — Analise Com CoALA

Explicar o que o framework ajudou a enxergar:

- mesma familia de memorias
- diferencas nas decisoes
- diferenca entre acao externa e acao interna
- progressao de complexidade: ciclo linear -> autoavaliacao -> busca em arvore

Tempo sugerido: `60s`

### Slide 8 — Conclusoes E Licoes Aprendidas

Fechar com uma conclusao pratica:

- para tarefas objetivas com ferramenta bem definida, `ReAct` ja resolve bem
- `Reflection` vale quando a primeira tentativa pode falhar
- `LATS` vale quando faz sentido explorar alternativas
- o maior ganho de confiabilidade veio de estruturar bem as ferramentas e os criterios de avaliacao

Tempo sugerido: `45-60s`

## Roteiro De Fala Em Uma Frase Por Slide

1. `Comparamos tres arquiteturas agenticas no mesmo benchmark oficial e usamos CoALA para organizar a analise.`
2. `A atividade pedia estudo, implementacao, comparacao empirica e apresentacao com demo.`
3. `Cada arquitetura muda a forma de decidir: agir em ciclo, refletir sobre erros ou explorar uma arvore de possibilidades.`
4. `No nosso projeto, isso virou codigo em Python com ferramentas, memoria persistente e versoes em LangGraph.`
5. `Usamos o benchmark oficial porque ele combina busca, calculo e julgamento final; aqui fazemos a demo.`
6. `Os tres agentes acertaram, mas os custos sao diferentes e isso muda quando cada arquitetura vale a pena.`
7. `O CoALA nos ajudou a separar memoria, acoes e estrategia de decisao de forma muito clara.`
8. `A principal licao foi que boa instrumentacao e boas ferramentas importam tanto quanto o prompt da arquitetura.`

## Demo Recomendada

Demo principal:

```bash
.venv/bin/python react_call.py
```

Backups:

```bash
.venv/bin/python reflection_call.py
.venv/bin/python lats_call.py
```

Se a API falhar ao vivo, usar imediatamente:

- `artifacts/graphs/react_langgraph_graph.png`
- `artifacts/graphs/reflection_langgraph_graph.png`
- `artifacts/graphs/lats_langgraph_graph.png`
- `artifacts/benchmark/comparison.md`

## Mensagem Final Que O Deck Deve Deixar

Arquitetura cognitiva nao e apenas "mais prompt". Ela muda o `mecanismo de decisao` do agente. No nosso projeto, isso apareceu de forma pratica: a tarefa foi a mesma, a resposta final tambem, mas custo, fluxo e interpretabilidade variaram bastante entre `ReAct`, `Reflection` e `LATS`.
