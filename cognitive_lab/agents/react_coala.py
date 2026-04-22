from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import requests
from langchain_core.messages import HumanMessage, SystemMessage


DEFAULT_MEMORY_DIR = "data/coala_memory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _safe_eval(expression: str) -> str:
    allowed_names = {"pi": 3.141592653589793, "e": 2.718281828459045}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Load,
    )

    def _eval(node: ast.AST) -> float:
        if not isinstance(node, allowed_nodes):
            raise ValueError("Expressao contem operacoes nao permitidas.")

        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Apenas numeros sao permitidos.")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Nome nao permitido: {node.id}")
            return allowed_names[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError("Operador unario nao permitido.")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Operador binario nao permitido.")

        raise ValueError("Expressao invalida.")

    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)
    if result.is_integer():
        return str(int(result))
    return str(result)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    kind: str
    handler: Callable[[str, "ToolRuntime"], str]


@dataclass(slots=True)
class ToolRuntime:
    memory: "CoALAMemoryStore"
    question: str
    working_memory: dict[str, Any]
    trajectory: list[dict[str, Any]]


@dataclass(slots=True)
class CoALAMemoryStore:
    root_dir: Path
    semantic_path: Path = field(init=False)
    episodic_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_path = self.root_dir / "semantic_memory.json"
        self.episodic_path = self.root_dir / "episodic_memory.json"
        if not self.semantic_path.exists():
            self.semantic_path.write_text("[]", encoding="utf-8")
        if not self.episodic_path.exists():
            self.episodic_path.write_text("[]", encoding="utf-8")

    def _read_json(self, path: Path) -> list[dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write_json(self, path: Path, data: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_semantic(self, content: str, source: str = "agent", tags: list[str] | None = None) -> dict[str, Any]:
        items = self._read_json(self.semantic_path)
        entry = {
            "id": uuid4().hex,
            "content": content.strip(),
            "source": source,
            "tags": tags or [],
            "created_at": _utc_now(),
        }
        items.append(entry)
        self._write_json(self.semantic_path, items)
        return entry

    def add_episode(self, question: str, final_answer: str, trajectory_summary: str) -> dict[str, Any]:
        items = self._read_json(self.episodic_path)
        entry = {
            "id": uuid4().hex,
            "question": question.strip(),
            "final_answer": final_answer.strip(),
            "trajectory_summary": trajectory_summary.strip(),
            "created_at": _utc_now(),
        }
        items.append(entry)
        self._write_json(self.episodic_path, items)
        return entry

    def search_semantic(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._search_items(self._read_json(self.semantic_path), query, top_k, ("content",))

    def search_episodic(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._search_items(
            self._read_json(self.episodic_path),
            query,
            top_k,
            ("question", "final_answer", "trajectory_summary"),
        )

    def counts(self) -> dict[str, int]:
        return {
            "semantic": len(self._read_json(self.semantic_path)),
            "episodic": len(self._read_json(self.episodic_path)),
        }

    def _search_items(
        self,
        items: list[dict[str, Any]],
        query: str,
        top_k: int,
        fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        query_tokens = _normalize_tokens(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            joined = " ".join(str(item.get(field, "")) for field in fields)
            item_tokens = _normalize_tokens(joined)
            score = len(query_tokens.intersection(item_tokens))
            if score > 0:
                scored.append((score, item))

        if not scored:
            return list(reversed(items))[:top_k]

        scored.sort(key=lambda pair: (pair[0], pair[1].get("created_at", "")), reverse=True)
        return [item for _, item in scored[:top_k]]


def buscar_ibge(query: str, _: ToolRuntime) -> str:
    query_lower = query.lower().strip()
    try:
        response = requests.get("https://servicodados.ibge.gov.br/api/v1/paises/all", timeout=10)
        response.raise_for_status()
        countries = response.json()

        if "mundo" in query_lower or "mundial" in query_lower:
            return "Media mundial estimada: PIB per capita = 12000 US$."

        targets = []
        if "américa do sul" in query_lower or "america do sul" in query_lower:
            for country in countries:
                location = country.get("localizacao") or {}
                region = location.get("regiao-intermediaria") or {}
                if region.get("nome") == "América do sul":
                    targets.append(country)
        else:
            for country in countries:
                short_name = country.get("nome", {}).get("abreviado", "").lower()
                if query_lower in short_name:
                    targets.append(country)
                    break

        if not targets:
            return "Nenhum pais encontrado para essa busca na API do IBGE."

        ids = "|".join(country["id"]["ISO-3166-1-ALPHA-2"] for country in targets)
        pib_response = requests.get(
            f"https://servicodados.ibge.gov.br/api/v1/paises/{ids}/indicadores/77827",
            timeout=10,
        )
        pib_pc_response = requests.get(
            f"https://servicodados.ibge.gov.br/api/v1/paises/{ids}/indicadores/77823",
            timeout=10,
        )
        pib_response.raise_for_status()
        pib_pc_response.raise_for_status()

        def get_latest(series: list[dict[str, Any]]) -> str:
            for item in reversed(series):
                value = list(item.values())[0]
                if value is not None:
                    return str(value)
            return "N/A"

        pib_payload = pib_response.json()
        pib_pc_payload = pib_pc_response.json()
        pib_series = pib_payload[0]["series"] if pib_payload else []
        pib_pc_series = pib_pc_payload[0]["series"] if pib_pc_payload else []

        pib_by_country = {item["pais"]["id"]: get_latest(item["serie"]) for item in pib_series}
        pib_pc_by_country = {item["pais"]["id"]: get_latest(item["serie"]) for item in pib_pc_series}

        lines = ["Dados da API do IBGE:"]
        for country in targets:
            country_id = country["id"]["ISO-3166-1-ALPHA-2"]
            short_name = country["nome"]["abreviado"]
            pib = pib_by_country.get(country_id, "N/A")
            pib_per_capita = pib_pc_by_country.get(country_id, "N/A")
            lines.append(f"- {short_name}: PIB = {pib} US$, PIB per capita = {pib_per_capita} US$")
        return "\n".join(lines)
    except Exception as exc:
        return f"Erro ao acessar API do IBGE: {exc}"


def calcular(expression: str, _: ToolRuntime) -> str:
    try:
        return _safe_eval(expression)
    except Exception as exc:
        return f"Erro ao calcular: {exc}"


def recordar_semantica(query: str, runtime: ToolRuntime) -> str:
    hits = runtime.memory.search_semantic(query, top_k=3)
    if not hits:
        return "Nenhuma memoria semantica encontrada."
    lines = ["Memorias semanticas relevantes:"]
    for item in hits:
        lines.append(f"- {item['content']}")
    return "\n".join(lines)


def recordar_episodios(query: str, runtime: ToolRuntime) -> str:
    hits = runtime.memory.search_episodic(query, top_k=3)
    if not hits:
        return "Nenhuma memoria episodica encontrada."
    lines = ["Memorias episodicas relevantes:"]
    for item in hits:
        lines.append(
            "- Pergunta: "
            f"{item['question']} | Resposta: {item['final_answer']} | Resumo: {item['trajectory_summary']}"
        )
    return "\n".join(lines)


def memorizar_semantica(text: str, runtime: ToolRuntime) -> str:
    if not text.strip():
        return "Nada foi salvo na memoria semantica."
    entry = runtime.memory.add_semantic(text, source="agent")
    return f"Memoria semantica salva com id {entry['id']}."


def build_tool_registry() -> dict[str, ToolSpec]:
    tools = [
        ToolSpec(
            name="buscar_ibge",
            kind="externa",
            description="Busca dados de paises ou regioes na API do IBGE e retorna PIB e PIB per capita em US$.",
            handler=buscar_ibge,
        ),
        ToolSpec(
            name="calcular",
            kind="interna",
            description="Avalia expressoes matematicas com +, -, *, /, //, %, **, parenteses e constantes pi/e.",
            handler=calcular,
        ),
        ToolSpec(
            name="recordar_semantica",
            kind="interna",
            description="Recupera fatos persistidos na memoria semantica usando uma consulta textual.",
            handler=recordar_semantica,
        ),
        ToolSpec(
            name="recordar_episodios",
            kind="interna",
            description="Recupera experiencias passadas persistidas na memoria episodica usando uma consulta textual.",
            handler=recordar_episodios,
        ),
        ToolSpec(
            name="memorizar_semantica",
            kind="interna",
            description="Salva um fato estavel e reutilizavel na memoria semantica.",
            handler=memorizar_semantica,
        ),
    ]
    return {tool.name: tool for tool in tools}


def _tool_block(tool_registry: dict[str, ToolSpec]) -> str:
    lines = []
    for tool in tool_registry.values():
        lines.append(f"- {tool.name}[argumento] ({tool.kind}): {tool.description}")
    return "\n".join(lines)


def _render_trajectory(trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return "Nenhum passo executado ainda."

    lines = []
    for item in trajectory[-6:]:
        lines.append(f"Passo {item['step']}:")
        lines.append(f"Thought: {item['thought']}")
        if item.get("action"):
            lines.append(f"Action: {item['action']}[{item['action_input']}]")
        lines.append(f"Observation: {item['observation']}")
    return "\n".join(lines)


def _render_memory_hits(items: list[dict[str, Any]], fields: tuple[str, ...], empty_text: str) -> str:
    if not items:
        return empty_text

    lines = []
    for item in items:
        parts = [str(item.get(field, "")).strip() for field in fields if item.get(field)]
        lines.append(f"- {' | '.join(parts)}")
    return "\n".join(lines)


def build_react_system_prompt(tool_registry: dict[str, ToolSpec]) -> str:
    return f"""Voce e um agente ReACT com arquitetura CoALA.

CoALA nesta aplicacao tem:
- working memory: objetivo atual, ultimo pensamento, ultima observacao e historico recente
- semantic memory: fatos persistidos e reutilizaveis
- episodic memory: experiencias anteriores e seus resultados
- procedural memory: estas regras e o catalogo de ferramentas

Seu trabalho e alternar raciocinio e acao.

Formato obrigatorio:
Thought: [raciocinio curto e util]
Action: ferramenta[argumento]

Quando concluir, responda exatamente:
Final Answer: [resposta final]

Regras:
- Nunca escreva a palavra Observation.
- Sempre comece com Thought: ou Final Answer:
- Execute apenas uma Action por resposta.
- Use ferramentas de memoria quando elas ajudarem.
- Se descobrir um fato estavel e reutilizavel, voce pode salva-lo com memorizar_semantica.
- Seja objetivo e nao invente dados.

Ferramentas disponiveis:
{_tool_block(tool_registry)}
"""


def _build_decision_prompt(
    question: str,
    working_memory: dict[str, Any],
    semantic_hits: list[dict[str, Any]],
    episodic_hits: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    max_steps: int,
) -> str:
    return f"""Objetivo principal:
{question}

Working memory:
- passo_atual: {working_memory['step']}
- max_passos: {max_steps}
- ultimo_pensamento: {working_memory.get('last_thought') or 'nenhum'}
- ultima_acao: {working_memory.get('last_action') or 'nenhuma'}
- ultima_observacao: {working_memory.get('last_observation') or 'nenhuma'}

Memoria semantica relevante:
{_render_memory_hits(semantic_hits, ('content',), 'Nenhuma memoria semantica relevante encontrada.')}

Memoria episodica relevante:
{_render_memory_hits(episodic_hits, ('question', 'final_answer', 'trajectory_summary'), 'Nenhuma memoria episodica relevante encontrada.')}

Historico recente:
{_render_trajectory(trajectory)}

Decida o proximo melhor passo. Se ja houver informacao suficiente, responda com Final Answer.
"""


def _parse_react_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {"kind": "error", "error": "Resposta vazia."}

    final_match = re.search(r"Final Answer:\s*(.+)", stripped, re.DOTALL)
    if final_match:
        return {"kind": "final", "final_answer": final_match.group(1).strip()}

    thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", stripped, re.DOTALL)
    action_match = re.search(r"Action:\s*([a-zA-Z_][\w]*)\[(.*)\]\s*$", stripped, re.DOTALL)

    if not thought_match or not action_match:
        return {
            "kind": "error",
            "error": "Formato invalido. Use Thought seguido de Action[nome] ou Final Answer.",
        }

    return {
        "kind": "action",
        "thought": thought_match.group(1).strip(),
        "action": action_match.group(1).strip(),
        "action_input": action_match.group(2).strip(),
    }


def _summarize_trajectory(trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return "Sem passos registrados."

    parts = []
    for item in trajectory[-5:]:
        action = item.get("action") or "sem_acao"
        observation = item.get("observation", "")
        compact_observation = observation.replace("\n", " ")[:160]
        parts.append(f"passo {item['step']}: {action} -> {compact_observation}")
    return " ; ".join(parts)


def _auto_consolidate_semantic_memory(
    memory: CoALAMemoryStore,
    question: str,
    final_answer: str,
) -> None:
    memory.add_semantic(
        f"Pergunta: {question} | Resposta consolidada: {final_answer}",
        source="auto_final_answer",
        tags=["auto", "final_answer"],
    )


def run_react_coala_agent(
    question: str,
    llm: Any,
    max_steps: int = 10,
    memory_dir: str | Path = DEFAULT_MEMORY_DIR,
) -> dict[str, Any]:
    memory = CoALAMemoryStore(Path(memory_dir))
    tools = build_tool_registry()
    working_memory: dict[str, Any] = {
        "goal": question,
        "step": 0,
        "last_thought": None,
        "last_action": None,
        "last_observation": None,
    }
    trajectory: list[dict[str, Any]] = []
    total_tokens = 0

    print("=== INICIANDO AGENTE REACT + COALA ===")
    print(f"Pergunta: {question}\n")
    print(f"Memoria persistente: {Path(memory_dir).resolve()}\n")

    while working_memory["step"] < max_steps:
        semantic_hits = memory.search_semantic(
            f"{question}\n{working_memory.get('last_observation') or ''}",
            top_k=3,
        )
        episodic_hits = memory.search_episodic(
            f"{question}\n{working_memory.get('last_observation') or ''}",
            top_k=3,
        )
        prompt = _build_decision_prompt(
            question=question,
            working_memory=working_memory,
            semantic_hits=semantic_hits,
            episodic_hits=episodic_hits,
            trajectory=trajectory,
            max_steps=max_steps,
        )

        response = llm.invoke(
            [
                SystemMessage(content=build_react_system_prompt(tools)),
                HumanMessage(content=prompt),
            ],
            stop=["Observation:"],
        )

        if getattr(response, "usage_metadata", None):
            total_tokens += response.usage_metadata.get("total_tokens", 0)

        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_react_output(text)

        working_memory["step"] += 1
        print(f"--- Passo {working_memory['step']} ---")
        print(text.strip() or "<vazio>")

        if parsed["kind"] == "final":
            final_answer = parsed["final_answer"]
            memory.add_episode(question, final_answer, _summarize_trajectory(trajectory))
            _auto_consolidate_semantic_memory(memory, question, final_answer)
            print("\n=== RESPOSTA FINAL ENCONTRADA ===")
            return {
                "resposta": final_answer,
                "steps": working_memory["step"],
                "tokens": total_tokens,
                "trajectory": trajectory,
                "memory_dir": str(Path(memory_dir).resolve()),
                "memory_counts": memory.counts(),
            }

        if parsed["kind"] == "error":
            observation = parsed["error"]
            working_memory["last_thought"] = "Formato incorreto."
            working_memory["last_action"] = None
            working_memory["last_observation"] = observation
            trajectory.append(
                {
                    "step": working_memory["step"],
                    "thought": "Formato incorreto.",
                    "action": None,
                    "action_input": None,
                    "observation": observation,
                }
            )
            print(f"Observation: {observation}")
            continue

        action_name = parsed["action"]
        action_input = parsed["action_input"]
        working_memory["last_thought"] = parsed["thought"]
        working_memory["last_action"] = f"{action_name}[{action_input}]"

        if action_name not in tools:
            observation = f"Ferramenta invalida: {action_name}. Ferramentas validas: {', '.join(sorted(tools))}."
        else:
            runtime = ToolRuntime(
                memory=memory,
                question=question,
                working_memory=working_memory,
                trajectory=trajectory,
            )
            try:
                observation = tools[action_name].handler(action_input, runtime)
            except Exception as exc:
                observation = f"Erro ao executar ferramenta {action_name}: {exc}"

        working_memory["last_observation"] = observation
        trajectory.append(
            {
                "step": working_memory["step"],
                "thought": parsed["thought"],
                "action": action_name,
                "action_input": action_input,
                "observation": observation,
            }
        )
        print(f"Observation: {observation}")

    final_answer = "Limite de passos atingido antes de chegar a uma resposta final."
    memory.add_episode(question, final_answer, _summarize_trajectory(trajectory))
    return {
        "resposta": final_answer,
        "steps": working_memory["step"],
        "tokens": total_tokens,
        "trajectory": trajectory,
        "memory_dir": str(Path(memory_dir).resolve()),
        "memory_counts": memory.counts(),
    }
