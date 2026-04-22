---
trigger: model_decision
description: cognitive-architectures-agent
---

---
name: cognitive-architectures-agent
description: >
  Implementa, compara e analisa agentes de IA usando arquiteturas cognitivas
  ReAct, Reflexion, LATS e CoALA. Use sempre que o usuário mencionar "ReAct",
  "Reflexion", "LATS", "CoALA", "agente cognitivo", "loop Thought-Action-Observation",
  "busca em árvore com LLM", ou quando pedir para construir, comparar ou analisar
  agentes que raciocinam e agem em múltiplos passos com ferramentas externas.
---

# Cognitive Architectures Agent Skill

Skill para implementar e comparar as 4 arquiteturas cognitivas para agentes de IA:
**ReAct**, **Reflexion**, **LATS** e **CoALA** — atividade prática das Aulas 4 & 5
de *Tópicos em Engenharia de Software — IA Agêntica*.

---

## As 4 Arquiteturas

| Arquitetura | Loop Principal | Memória (CoALA) | Complexidade |
|-------------|---------------|-----------------|--------------|
| **ReAct** | Thought → Action → Observation → repete | Working | Baixa |
| **Reflexion** | ReAct + reflexão sobre erros + retry | Working + Episodic | Média |
| **LATS** | Busca em árvore Monte Carlo + LLM-juiz | Working + planejamento | Alta |
| **CoALA** | Framework conceitual (não implementado diretamente) | Todos | N/A |

ReAct é a base. Reflexion adiciona autorreflexão. LATS adiciona exploração de
múltiplos caminhos. CoALA é o mapa teórico que classifica todas.

---

## APIs Gratuitas (formato OpenAI-compatível)

```python
from openai import OpenAI

# Gemini — 1.000 req/dia, sem cartão
client = OpenAI(api_key="SUA_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# OU Groq — 1.000 req/dia, sem cartão
client = OpenAI(api_key="SUA_KEY", base_url="https://api.groq.com/openai/v1")

# Modelo: "gemini-2.5-flash-lite-preview-06-17" ou "llama-3.3-70b-versatile"
```

Para trocar de provedor: mude apenas `base_url` e `api_key`.

---

## Implementação: ReAct (Obrigatório)

```python
import re

SYSTEM_PROMPT = """Resolva problemas alternando entre raciocínio e ação.

Formato obrigatório em cada passo:
Thought: [seu raciocínio]
Action: ferramenta[argumento]

Ferramentas disponíveis:
- buscar[query]: busca informação
- calcular[expressão]: avalia matemática

Quando tiver a resposta:
Final Answer: [resposta]"""

FERRAMENTAS = {
    "buscar": lambda q: f"[Resultado de busca para '{q}']",  # substitua por API real
    "calcular": lambda e: str(eval(e)) if e else "erro",
}

def react_agent(pergunta: str, max_steps: int = 10) -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pergunta}]
    tokens, steps = 0, 0

    for _ in range(max_steps):
        steps += 1
        resp = client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=msgs, stop=["Observation:"]
        )
        tokens += resp.usage.total_tokens
        texto = resp.choices[0].message.content
        msgs.append({"role": "assistant", "content": texto})

        if "Final Answer:" in texto:
            return {"resposta": texto.split("Final Answer:")[-1].strip(),
                    "steps": steps, "tokens": tokens}

        m = re.search(r'Action:\s*(\w+)\[(.+?)\]', texto)
        obs = FERRAMENTAS[m.group(1)](m.group(2)) if m and m.group(1) in FERRAMENTAS \
              else "Ferramenta inválida."
        msgs.append({"role": "user", "content": f"Observation: {obs}"})

    return {"resposta": "Limite atingido.", "steps": steps, "tokens": tokens}
```

---

## Implementação: Reflexion

Reflexion é ReAct com loop externo de autorreflexão. Se a resposta for
insatisfatória, o agente reflete sobre o erro e tenta novamente com esse
aprendizado como memória episódica.

```python
REFLEXION_SYSTEM = """Analise a tentativa anterior e identifique o que deu errado.
Seja específico: cite ferramentas mal usadas ou raciocínios incorretos.
Formato: 'Errei porque [X]. Na próxima tentativa vou [Y].'"""

def avaliar(resposta: str) -> bool:
    """Critério simples — adapte ao seu problema."""
    return any(c.isdigit() for c in resposta) and len(resposta) > 10

def reflexion_agent(pergunta: str, max_tentativas: int = 3) -> dict:
    reflexoes, total_tokens, total_steps = [], 0, 0

    for tentativa in range(1, max_tentativas + 1):
        contexto = ""
        if reflexoes:
            contexto = "\n\n[Tentativas anteriores]\n" + \
                       "\n".join(f"T{i+1}: {r}" for i, r in enumerate(reflexoes))

        resultado = react_agent(pergunta + contexto)
        total_tokens += resultado["tokens"]
        total_steps += resultado["steps"]

        if avaliar(resultado["resposta"]):
            return {"resposta": resultado["resposta"], "tentativas": tentativa,
                    "reflexoes": reflexoes, "tokens": total_tokens}

        r = client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=[{"role": "system", "content": REFLEXION_SYSTEM},
                      {"role": "user", "content":
                       f"Pergunta: {pergunta}\nResposta: {resultado['resposta']}"}]
        )
        total_tokens += r.usage.total_tokens
        reflexoes.append(r.choices[0].message.content)

    return {"resposta": resultado["resposta"], "tentativas": max_tentativas,
            "reflexoes": reflexoes, "tokens": total_tokens}
```

---

## Implementação: LATS (Desafiador — opcional)

LATS expande múltiplos caminhos na árvore, avalia cada um com um LLM-juiz e
escolhe o melhor para continuar. Muito mais chamadas à API — use Groq ou
adicione `time.sleep(1)` para não estourar rate limits.

```python
import math
from dataclasses import dataclass, field

@dataclass
class Node:
    state: list
    value: float = 0.0
    visits: int = 0
    children: list = field(default_factory=list)
    parent: object = None

    def ucb(self, c=1.4) -> float:
        if self.visits == 0: return float('inf')
        return self.value/self.visits + c*math.sqrt(math.log(self.parent.visits)/self.visits)

def avaliar_no(node: Node, pergunta: str) -> float:
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in node.state[-3:])
    r = client.chat.completions.create(
        model="gemini-2.5-flash-lite-preview-06-17",
        messages=[{"role": "user", "content":
                   f"Pergunta: {pergunta}\nHistórico:\n{hist}\n"
                   "Progresso de 0 a 10 (só o número):"}], max_tokens=5)
    try: return int(r.choices[0].message.content.strip()) / 10.0
    except: return 0.5

def lats_agent(pergunta: str, expansoes: int = 3, profundidade: int = 5) -> dict:
    raiz = Node(state=[{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": pergunta}], visits=1)
    total_tokens = 0

    for _ in range(profundidade):
        no = raiz
        while no.children:
            no = max(no.children, key=lambda n: n.ucb())

        for _ in range(expansoes):
            r = client.chat.completions.create(
                model="gemini-2.5-flash-lite-preview-06-17",
                messages=no.state, stop=["Observation:"], temperature=0.7)
            total_tokens += r.usage.total_tokens
            texto = r.choices[0].message.content

            if "Final Answer:" in texto:
                return {"resposta": texto.split("Final Answer:")[-1].strip(),
                        "tokens": total_tokens}

            m = re.search(r'Action:\s*(\w+)\[(.+?)\]', texto)
            obs = FERRAMENTAS[m.group(1)](m.group(2)) if m and m.group(1) in FERRAMENTAS \
                  else "Ferramenta inválida."
            filho = Node(state=no.state + [{"role": "assistant", "content": texto},
                                           {"role": "user", "content": f"Observation: {obs}"}],
                         parent=no)
            filho.value = avaliar_no(filho, pergunta)
            filho.visits = 1
            no.children.append(filho)

        melhor = max(no.children, key=lambda n: n.value)
        no.value += melhor.value
        no.visits += 1

    return {"resposta": "Não convergiu.", "tokens": total_tokens}
```

---

## Problema de Teste Recomendado

Use o mesmo problema em todas as arquiteturas para comparação justa:

> **"Pesquise os 3 países com maior PIB da América do Sul, calcule a média do
> PIB per capita deles e responda: essa média é maior ou menor que a média mundial?"**

---

## Análise Comparativa (CoALA)

| Critério | ReAct | Reflexion | LATS |
|----------|-------|-----------|------|
| Resposta correta? | | | |
| Chamadas ao LLM | | | |
| Tempo total (s) | | | |
| Tokens consumidos | | | |
| Memória (CoALA) | Working | Working + Episodic | Working |
| Quando usar? | Tarefas simples | Quando erra 1ª vez | Decisões críticas |

**Classificação CoALA:** Working memory = contexto atual (todas). Episodic memory =
histórico de erros (Reflexion). Ações internas = raciocínio/reflexão/juiz.
Ações externas = ferramentas. Decisão: ReAct greedy | Reflexion retry | LATS planejamento.

---

## Dicas

- Comece com ReAct + 1 ferramenta. Só adicione complexidade depois que funcionar.
- O `react_agent()` é reutilizado dentro do `reflexion_agent()` — não reescreva.
- Meça tokens e tempo desde o início: são os dados da análise comparativa.
- LATS faz muitas chamadas — prefira Groq (30 RPM) ou adicione `time.sleep(1)`.

---

## Referências

- ReAct: arXiv:2210.03629 | Reflexion: arXiv:2303.11366
- LATS: arXiv:2310.04406 | CoALA: arXiv:2309.02427
- NotebookLM: notebooklm.google.com/notebook/8aefbec6-7eaa-4540-9ef8-eae1650118ef
- Anthropic: anthropic.com/research/building-effective-agents
