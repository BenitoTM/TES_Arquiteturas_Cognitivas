# Especificacao Completa Dos Slides

Este arquivo foi feito para servir de `fonte unica` para outra IA gerar um `pptx` ou `pdf` da apresentacao. O idioma da apresentacao deve ser `portugues do Brasil`.

## 1. Objetivo Do Deck

Gerar uma apresentacao academica curta, clara e visualmente limpa para a disciplina, mostrando:

- o que a atividade do professor pediu
- o que foi implementado no repositório
- como as arquiteturas diferem
- o benchmark oficial usado
- a comparacao empirica
- a conclusao do grupo

Meta de duracao: `7-8 minutos`.

## 2. Fonte Da Verdade

Usar esta ordem de prioridade para o conteudo:

1. PDF da atividade: `/Users/benito/Downloads/aula4_5_arquiteturas_cognitivas (2).pdf`
2. Benchmark consolidado:
   - `artifacts/benchmark/comparison.json`
   - `artifacts/benchmark/comparison.md`
3. Contexto tecnico do projeto:
   - `readme.md`
   - `compare_agents.py`
   - `cognitive_lab/agents/react_coala.py`
   - `cognitive_lab/agents/reflection.py`
   - `cognitive_lab/agents/lats.py`
   - `cognitive_lab/runtime/portkey.py`
4. Apoio narrativo existente:
   - `docs/analise_comparativa.md`
   - `docs/roteiro_apresentacao.md`

## 3. Regras Importantes Para Nao Errar

- Nao dizer que o projeto implementou `4 agentes completos`.
- Dizer corretamente:
  - `3 arquiteturas implementadas`: ReAct, Reflection/Reflexion e LATS
  - `CoALA` foi usado como framework conceitual e organizacao de memoria
- Nao inventar resultados alem dos artefatos.
- Nao usar os tempos antigos que aparecem em `docs/analise_comparativa.md`; usar os valores mais recentes de `artifacts/benchmark/comparison.*`.
- Nao transformar o deck em aula teorica longa sobre os papers; a prioridade e mostrar o que o grupo estudou, implementou e comparou.

## 4. Direcao Visual Recomendada

Estilo:

- academico-tecnico
- limpo
- moderno
- com cara de demo de engenharia, nao de marketing

Paleta sugerida:

- fundo claro `#F7F6F2`
- texto principal `#1F2937`
- azul tecnico `#2563EB`
- verde de destaque `#059669`
- laranja de alerta/custo `#EA580C`
- cinza de apoio `#6B7280`

Tipografia sugerida:

- titulos: `Poppins SemiBold`
- corpo: `Lato`
- codigo/comandos: `JetBrains Mono` ou equivalente

Elementos visuais:

- usar cards e diagramas simples
- incluir pelo menos 1 slide com comparacao em tabela
- incluir 1 slide com visual dos grafos LangGraph
- evitar excesso de texto por slide

## 5. Estrutura Final Recomendada

Produzir `8 slides principais` e, se houver espaco, `1 ou 2 slides backup` no final.

## 6. Conteudo Detalhado Slide A Slide

### Slide 1 — Capa

Titulo:

`Arquiteturas Cognitivas Para Agentes de IA`

Subtitulo:

`Comparacao pratica entre ReAct, Reflection/Reflexion e LATS usando CoALA como lente de analise`

Texto curto de apoio:

- disciplina: `Topicos em Engenharia de Software`
- atividade: `Aulas 4 e 5`
- tipo: `pesquisa + implementacao + apresentacao`

Objetivo do slide:

- situar o tema
- mostrar que ha parte pratica e comparativa

Layout sugerido:

- titulo grande na esquerda
- 3 cards ou 3 etiquetas na direita: `ReAct`, `Reflection`, `LATS`
- linha menor abaixo com `CoALA = framework conceitual`

Notas do apresentador:

`Nesta atividade, nosso grupo estudou arquiteturas cognitivas para agentes e implementou tres delas sobre o mesmo benchmark para comparar comportamento, custo e interpretabilidade.`

### Slide 2 — O Que A Atividade Pedia

Titulo:

`O Que O Professor Pediu`

Conteudo on-slide:

- estudar as arquiteturas com base nos papers e no NotebookLM
- implementar pelo menos `ReAct + 1` arquitetura adicional
- comparar todas no mesmo problema
- usar `CoALA` para classificar memoria, acoes e decisao
- apresentar resultados em `7-8 minutos` com `demo obrigatoria`

Box lateral com a pergunta oficial:

`Pesquise os 3 paises com maior PIB da America do Sul, calcule a media do PIB per capita deles e responda: essa media e maior ou menor que a media mundial?`

Objetivo do slide:

- conectar a entrega do grupo ao enunciado

Layout sugerido:

- esquerda com lista dos requisitos
- direita com um card destacado contendo a pergunta oficial

Notas do apresentador:

`O ponto central da tarefa nao era apenas ler os papers, mas transformar isso em agentes reais, comparar os resultados e explicar os trade-offs com a lente do CoALA.`

### Slide 3 — Arquiteturas E Papel Do CoALA

Titulo:

`Arquiteturas Comparadas`

Conteudo on-slide em 4 blocos:

- `ReAct`
  - ciclo `Thought -> Action -> Observation`
  - bom para uso de ferramentas em fluxo linear
- `Reflection / Reflexion`
  - adiciona `julgamento` e `reflexao`
  - tenta de novo com aprendizado verbal
- `LATS`
  - explora `multiplos ramos`
  - escolhe o caminho mais promissor
- `CoALA`
  - organiza `working`, `episodic`, `semantic` e `procedural memory`
  - foi usado como `framework conceitual`, nao como quarto agente independente

Rodape pequeno com referencias dos papers:

- `Yao et al.`
- `Shinn et al.`
- `Zhou et al.`
- `Sumers et al.`

Objetivo do slide:

- explicar em linguagem simples o papel de cada arquitetura

Layout sugerido:

- 4 colunas curtas
- CoALA destacado com cor diferente, para deixar claro que ele classifica as outras arquiteturas

Notas do apresentador:

`ReAct foi a base, Reflection adicionou uma camada meta-cognitiva de revisao, LATS trouxe planejamento por busca em arvore, e o CoALA nos ajudou a organizar tudo isso em termos de memoria e decisao.`

### Slide 4 — Como O Projeto Foi Implementado

Titulo:

`Como Transformamos Isso Em Codigo`

Conteudo on-slide:

- stack principal:
  - `Python 3.13`
  - `Portkey`
  - `LangChain`
  - `LangGraph`
- modelo padrao do projeto:
  - `z-ai/glm-4.5-air:free`
- ferramentas usadas pelos agentes:
  - busca de dados do benchmark
  - calculadora
  - recuperacao de memoria
- memoria persistente em JSON:
  - `semantic`
  - `episodic`
  - `reflection` quando aplicavel
- artefatos gerados:
  - grafos em `artifacts/graphs`
  - benchmark em `artifacts/benchmark`

Conteudo visual recomendado:

- usar as imagens:
  - `artifacts/graphs/react_langgraph_graph.png`
  - `artifacts/graphs/reflection_langgraph_graph.png`
  - `artifacts/graphs/lats_langgraph_graph.png`

Objetivo do slide:

- mostrar que o trabalho foi implementado de verdade no repositório

Layout sugerido:

- metade esquerda: stack e componentes
- metade direita: miniaturas dos 3 grafos

Notas do apresentador:

`O projeto nao ficou so na teoria. Nos implementamos os agentes, salvamos memoria persistente, geramos artefatos de benchmark e ainda produzimos versoes em LangGraph para visualizar o fluxo de cada arquitetura.`

### Slide 5 — Benchmark E Demo

Titulo:

`Benchmark Oficial E Demo`

Conteudo on-slide:

- problema unico para todas as arquiteturas
- combina `tool use`, `calculo` e `julgamento`
- ideal para comparar custo e estrategia de decisao

Resposta de referencia do benchmark:

- top 3 por PIB: `Brasil`, `Argentina`, `Venezuela`
- media do PIB per capita do top 3: `13475.32 US$`
- media mundial do PIB per capita: `13631.20 US$`
- comparacao: `menor`

Bloco de demo:

- comando principal:

```bash
.venv/bin/python react_call.py
```

- backups:

```bash
.venv/bin/python reflection_call.py
.venv/bin/python lats_call.py
```

Mensagem curta para aparecer no slide:

`Escolhemos demonstrar o ReAct ao vivo por ser a execucao mais curta e mais facil de acompanhar em 1-2 minutos.`

Objetivo do slide:

- introduzir a demo ao vivo
- justificar a escolha da execucao mostrada

Layout sugerido:

- pergunta do benchmark no topo
- bloco de comando no centro
- rodape com a mensagem de justificativa

Notas do apresentador:

`O benchmark foi o mesmo para todas as arquiteturas. Isso foi importante para manter a comparacao justa. Na apresentacao, a demo ao vivo pode focar no ReAct, e os demais resultados podem ser mostrados com base nos artefatos salvos.`

### Slide 6 — Resultados Comparativos

Titulo:

`Resultados Empiricos`

Usar exatamente estes dados:

| Arquitetura | Correta? | Chamadas ao LLM | Tempo total (s) | Tokens | Steps | Memoria CoALA | Complexidade | Quando usar? |
|---|---:|---:|---:|---:|---:|---|---|---|
| ReAct + CoALA | Sim | 5 | 12.1956 | 6727 | 5 | working + semantic + episodic + procedural | media | quando a tarefa depende de ferramentas externas e ciclo Thought -> Action -> Observation |
| Reflection / Reflexion | Sim | 6 | 13.8151 | 8403 | 6 | working + semantic + episodic + procedural + reflection memory | alta | quando vale pagar mais chamadas ao LLM para revisar e corrigir tentativas |
| LATS + CoALA | Sim | 8 | 24.4531 | 10515 | 4 | working + semantic + episodic + procedural + arvore de busca | alta | quando compensa explorar multiplos ramos e selecionar a trajetoria mais promissora |

Mensagem principal do slide:

- todas acertaram
- `ReAct` foi o mais eficiente
- `Reflection` custou um pouco mais para ganhar revisao
- `LATS` foi o mais caro e o mais deliberativo

Objetivo do slide:

- mostrar a parte empirica da atividade

Layout sugerido:

- tabela principal ocupando quase todo o slide
- 3 pequenos destaques no topo ou lateral:
  - `Mais rapido: ReAct`
  - `Mais robusto conceitualmente: Reflection`
  - `Maior exploracao: LATS`

Notas do apresentador:

`O ponto importante aqui e que a resposta final foi correta em todos os casos. O que muda e o caminho ate ela: quantidade de chamadas ao modelo, consumo de tokens, tempo total e complexidade da estrategia de decisao.`

### Slide 7 — Leitura Pelo CoALA

Titulo:

`O Que O CoALA Nos Ajudou A Enxergar`

Conteudo on-slide:

- `working memory`
  - estado atual da execucao
- `episodic memory`
  - historico de execucoes anteriores
- `semantic memory`
  - fatos consolidados e reutilizaveis
- `procedural memory`
  - regras, prompts e fluxo de controle

Segunda linha do slide:

- `ReAct`: decisao iterativa local
- `Reflection`: decisao com autoavaliacao e nova tentativa
- `LATS`: decisao com planejamento por busca em arvore

Texto de sintese:

`CoALA foi util porque separou memoria, acao e estrategia de decisao de forma muito clara na comparacao.`

Objetivo do slide:

- fazer a ponte entre teoria e implementacao

Layout sugerido:

- parte superior com 4 cards de memoria
- parte inferior com uma linha de progressao:
  - `linear`
  - `reflexiva`
  - `deliberativa`

Notas do apresentador:

`CoALA nao foi so um nome citado no trabalho. Ele realmente ajudou a organizar a analise: quais memorias cada agente usa, quando a acao e externa ou interna, e como a tomada de decisao fica mais sofisticada de ReAct para Reflection e depois para LATS.`

### Slide 8 — Conclusoes

Titulo:

`Conclusoes E Licoes Aprendidas`

Conteudo on-slide:

- `ReAct` ja resolve bem quando a tarefa e objetiva e as ferramentas sao confiaveis
- `Reflection` faz sentido quando errar na primeira tentativa e provavel e corrigir vale o custo
- `LATS` faz sentido quando explorar alternativas realmente pode melhorar a resposta
- estruturar bem as ferramentas e os criterios de avaliacao foi tao importante quanto escolher a arquitetura

Fechamento sugerido:

`Mesma tarefa, mesma resposta final, mecanismos de decisao diferentes.`

Objetivo do slide:

- fechar com mensagem pratica e memoravel

Layout sugerido:

- 4 conclusoes em cards
- frase final destacada no rodape

Notas do apresentador:

`Se a tarefa for bem estruturada, ReAct pode ser suficiente. Quando o espaco de erro cresce, Reflection e LATS passam a fazer mais sentido. A principal licao do projeto foi que arquitetura cognitiva muda o modo de decidir, e nao apenas o texto do prompt.`

## 7. Slides Backup Opcionais

### Backup A — Estrutura Do Repositório

Usar se houver pergunta tecnica da turma ou do professor.

Conteudo sugerido:

- `compare_agents.py`
- `react_call.py`
- `reflection_call.py`
- `lats_call.py`
- `cognitive_lab/agents/react_coala.py`
- `cognitive_lab/agents/reflection.py`
- `cognitive_lab/agents/lats.py`
- `artifacts/benchmark/comparison.*`

### Backup B — Grafo Das Arquiteturas

Usar as imagens:

- `artifacts/graphs/react_langgraph_graph.png`
- `artifacts/graphs/reflection_langgraph_graph.png`
- `artifacts/graphs/lats_langgraph_graph.png`

Objetivo:

- responder visualmente como o fluxo muda entre as arquiteturas

## 8. Assets Que Podem Ser Usados Diretamente

Imagens:

- `artifacts/graphs/react_langgraph_graph.png`
- `artifacts/graphs/reflection_langgraph_graph.png`
- `artifacts/graphs/lats_langgraph_graph.png`

Textos e dados:

- `artifacts/benchmark/comparison.md`
- `artifacts/benchmark/comparison.json`
- `docs/analise_comparativa.md`
- `docs/roteiro_apresentacao.md`
- `readme.md`

## 9. Resposta De Referencia Do Benchmark

Se a IA que gerar o deck quiser mencionar explicitamente a resposta esperada do problema, usar exatamente:

- `Brasil`, `Argentina` e `Venezuela`
- media do PIB per capita do top 3: `13475.32 US$`
- media mundial do PIB per capita: `13631.20 US$`
- conclusao: a media do top 3 e `menor` que a media mundial

Fonte:

- `artifacts/benchmark/comparison.md`
- `artifacts/benchmark/comparison.json`

## 10. Frases Prontas Que Podem Entrar No Deck

- `Implementamos tres arquiteturas agenticas sobre o mesmo benchmark oficial e usamos CoALA como lente de analise.`
- `A principal diferenca entre elas nao foi a resposta final, mas o mecanismo de decisao para chegar nela.`
- `ReAct foi o mais simples e eficiente; Reflection trouxe revisao iterativa; LATS trouxe busca deliberativa com maior custo.`
- `CoALA ajudou a separar memoria operacional, memoria persistente e estrategia de decisao.`

## 11. O Que Evitar No Deck

- Nao lotar os slides com definicoes longas de paper.
- Nao usar texto corrido extenso.
- Nao dizer que `CoALA` foi um quarto agente executado no benchmark.
- Nao usar numeros diferentes dos artefatos em `artifacts/benchmark`.
- Nao inventar `complexidade em linhas` como se fosse uma medicao oficial do projeto; nos artefatos a complexidade principal esta registrada de forma qualitativa (`media` ou `alta`).
- Nao gastar muito tempo com detalhes de instalacao.

## 12. Se Outra IA For Gerar O PPTX Automaticamente

Ela deve:

- produzir um deck de `8 slides principais`
- manter o idioma em `portugues`
- usar os dados reais da tabela acima
- incluir referencias visuais aos grafos do repositório
- deixar espaco para a demo ao vivo
- priorizar clareza, comparacao e conclusao pratica
