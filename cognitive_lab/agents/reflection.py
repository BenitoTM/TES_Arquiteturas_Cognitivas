from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from cognitive_lab.agents import react_coala
from cognitive_lab.runtime.pricing import add_token_usage, zero_token_usage


DEFAULT_REFLECTION_MEMORY_DIR = "data/reflection_memory"


@dataclass(slots=True)
class ReflectionMemoryStore:
    root_dir: Path
    reflections_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.reflections_path = self.root_dir / "reflections.json"
        if not self.reflections_path.exists():
            self.reflections_path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.reflections_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write(self, data: list[dict[str, Any]]) -> None:
        self.reflections_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_reflection(
        self,
        question: str,
        reflection: str,
        feedback: str,
        trajectory_summary: str,
    ) -> dict[str, Any]:
        items = self._read()
        entry = {
            "id": uuid4().hex,
            "question": question.strip(),
            "reflection": reflection.strip(),
            "feedback": feedback.strip(),
            "trajectory_summary": trajectory_summary.strip(),
            "created_at": react_coala._utc_now(),
        }
        items.append(entry)
        self._write(items)
        return entry

    def search_reflections(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = react_coala._normalize_tokens(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self._read():
            joined = " ".join(
                [
                    item.get("question", ""),
                    item.get("reflection", ""),
                    item.get("feedback", ""),
                    item.get("trajectory_summary", ""),
                ]
            )
            score = len(query_tokens.intersection(react_coala._normalize_tokens(joined)))
            if score > 0:
                scored.append((score, item))

        if not scored:
            return list(reversed(self._read()))[:top_k]

        scored.sort(key=lambda pair: (pair[0], pair[1].get("created_at", "")), reverse=True)
        return [item for _, item in scored[:top_k]]

    def count(self) -> int:
        return len(self._read())


def build_reflection_actor_system_prompt(tool_registry: dict[str, react_coala.ToolSpec]) -> str:
    return f"""Voce e um agente Reflection (Reflexion-style).

Seu objetivo e resolver a tarefa com tentativa, avaliacao e melhoria.
Use reflexoes anteriores para evitar repetir erros.

Formato obrigatorio:
Thought: [raciocinio curto e util]
Action: ferramenta[argumento]

Quando tiver uma resposta candidata completa, responda exatamente:
Final Answer: [resposta final]

Regras:
- Nunca escreva a palavra Observation.
- Execute apenas uma Action por resposta.
- Use as reflexoes anteriores para corrigir sua estrategia.
- Se o historico ja tiver dados suficientes, pare de buscar e feche a resposta.
- Se a ultima observacao for True ou False apos uma comparacao, converta isso em maior/menor e responda Final Answer.
- Para o benchmark oficial, voce so pode encerrar quando tiver:
  1. os 3 paises corretos;
  2. a media do PIB per capita;
  3. a comparacao com a media mundial;
  4. uma resposta final clara.
- Para o benchmark de diferenca absoluta, use analisar_benchmark_top3_diferenca[] e so encerre quando tiver:
  1. os 3 paises corretos;
  2. a media do PIB per capita do top 3;
  3. a media mundial;
  4. a diferenca em US$;
  5. a comparacao com 1000 US$.
- Nao repita a mesma acao sem progresso.
- Seja objetivo e nao invente dados.

Ferramentas disponiveis:
{react_coala._tool_block(tool_registry)}
"""


def build_reflection_attempt_prompt(
    question: str,
    attempt_number: int,
    max_attempts: int,
    max_steps: int,
    working_memory: dict[str, Any],
    reflections: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    episodic_hits: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> str:
    reflection_block = (
        "\n".join(f"- {item['reflection']} | feedback anterior: {item['feedback']}" for item in reflections)
        if reflections
        else "Nenhuma reflexao anterior relevante."
    )

    return f"""Objetivo principal:
{question}

Tentativa atual: {attempt_number} de {max_attempts}

Working memory:
- passo_atual: {working_memory['step']}
- max_passos: {max_steps}
- ultimo_pensamento: {working_memory.get('last_thought') or 'nenhum'}
- ultima_acao: {working_memory.get('last_action') or 'nenhuma'}
- ultima_observacao: {working_memory.get('last_observation') or 'nenhuma'}

Reflexoes anteriores:
{reflection_block}

Memoria semantica relevante:
{react_coala._render_memory_hits(semantic_hits, ('content',), 'Nenhuma memoria semantica relevante encontrada.')}

Memoria episodica relevante:
{react_coala._render_memory_hits(episodic_hits, ('question', 'final_answer', 'trajectory_summary'), 'Nenhuma memoria episodica relevante encontrada.')}

Historico recente desta tentativa:
{react_coala._render_trajectory(trajectory)}

Instrucao operacional:
- Se ja houver TOP_3_PIB_AMERICA_DO_SUL e MEDIA_MUNDIAL_PIB_PER_CAPITA nas observacoes, faca no maximo uma acao de calculo e depois responda Final Answer.
- Se ja houver BENCHMARK_TOP3_DIFERENCA_ABSOLUTA nas observacoes, responda Final Answer sem fazer novas buscas.
- Se a ultima observacao for True ou False, responda Final Answer imediatamente.
- Se ja houver numeros suficientes, nao faca nova busca.
- Se a resposta ja estiver pronta, responda Final Answer imediatamente.

Produza o proximo passo.
"""


def build_reflection_judge_prompt(question: str, final_answer: str | None, trajectory: list[dict[str, Any]]) -> str:
    return f"""Voce e um avaliador rigoroso de uma tentativa de agente.

Pergunta original:
{question}

Resposta candidata:
{final_answer or 'Nenhuma resposta final foi produzida.'}

Trajetoria resumida:
{react_coala._summarize_trajectory(trajectory)}

Avalie explicitamente estes itens:
- os tres paises corretos
- a media calculada
- a comparacao com a media mundial
- a clareza da resposta final

Responda exatamente neste formato:
Verdict: ACCEPT ou RETRY
Feedback: [diga objetivamente quais itens faltaram ou o que deve ser corrigido, e qual deve ser a proxima acao]

Use ACCEPT apenas se os quatro itens estiverem satisfeitos.
"""


def build_reflection_prompt(
    question: str,
    attempt_result: dict[str, Any],
    feedback: str,
) -> str:
    return f"""Voce e um refletor que ajuda um agente a melhorar na proxima tentativa.

Pergunta:
{question}

Resultado da tentativa:
- resposta_final: {attempt_result.get('final_answer') or 'nenhuma'}
- resumo_da_trajetoria: {react_coala._summarize_trajectory(attempt_result.get('trajectory', []))}

Feedback do avaliador:
{feedback}

Escreva exatamente:
Reflection: [2 ou 3 frases curtas, operacionais e acionaveis, dizendo o que fazer na proxima tentativa, o que evitar repetir e quando encerrar com Final Answer]
"""


def parse_judge_output(text: str) -> dict[str, str]:
    verdict_match = re.search(r"Verdict:\s*(ACCEPT|RETRY)", text)
    feedback_match = re.search(r"Feedback:\s*(.+)", text, re.DOTALL)
    if not verdict_match:
        return {"verdict": "RETRY", "feedback": "O avaliador retornou um formato invalido."}
    return {
        "verdict": verdict_match.group(1).strip(),
        "feedback": feedback_match.group(1).strip() if feedback_match else "Sem feedback detalhado.",
    }


def parse_reflection_output(text: str) -> str:
    match = re.search(r"Reflection:\s*(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() or "Use os dados ja coletados, evite repetir buscas e encerre com Final Answer assim que a comparacao estiver pronta."


def _is_repeated_action_without_progress(
    trajectory: list[dict[str, Any]],
    action_name: str,
    action_input: str,
) -> bool:
    if not trajectory:
        return False
    last_step = trajectory[-1]
    return (
        last_step.get("action") == action_name
        and last_step.get("action_input") == action_input
    )


def run_reflection_attempt(
    question: str,
    llm: Any,
    reflections: list[dict[str, Any]],
    attempt_number: int,
    max_attempts: int,
    max_steps: int,
    memory_dir: str | Path,
) -> dict[str, Any]:
    memory = react_coala.CoALAMemoryStore(Path(memory_dir))
    tools = react_coala.build_tool_registry()
    working_memory: dict[str, Any] = {
        "goal": question,
        "step": 0,
        "last_thought": None,
        "last_action": None,
        "last_observation": None,
    }
    trajectory: list[dict[str, Any]] = []
    total_tokens = 0
    llm_calls = 0
    token_usage = zero_token_usage()

    # Loop principal do Reflection: agir com base nas reflexoes anteriores.
    for _ in range(max_steps):
        semantic_hits = memory.search_semantic(f"{question}\n{working_memory.get('last_observation') or ''}", top_k=3)
        episodic_hits = memory.search_episodic(f"{question}\n{working_memory.get('last_observation') or ''}", top_k=3)
        prompt = build_reflection_attempt_prompt(
            question=question,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            max_steps=max_steps,
            working_memory=working_memory,
            reflections=reflections,
            semantic_hits=semantic_hits,
            episodic_hits=episodic_hits,
            trajectory=trajectory,
        )

        llm_calls += 1
        response = llm.invoke(
            [
                SystemMessage(content=build_reflection_actor_system_prompt(tools)),
                HumanMessage(content=prompt),
            ],
            stop=["Observation:"],
        )
        if getattr(response, "usage_metadata", None):
            total_tokens += response.usage_metadata.get("total_tokens", 0)
            token_usage = add_token_usage(token_usage, response.usage_metadata)

        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = react_coala._parse_react_output(text)
        working_memory["step"] += 1

        print(f"--- Tentativa {attempt_number} | Passo {working_memory['step']} ---")
        print(text.strip() or "<vazio>")

        if parsed["kind"] == "final":
            return {
                "status": "final",
                "final_answer": parsed["final_answer"],
                "steps": working_memory["step"],
                "tokens": total_tokens,
                "token_usage": token_usage,
                "llm_calls": llm_calls,
                "trajectory": trajectory,
                "working_memory": working_memory,
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

        if _is_repeated_action_without_progress(trajectory, action_name, action_input):
            observation = "Acao repetida sem progresso. Escolha outra ferramenta ou finalize se os dados ja forem suficientes."
        elif action_name not in tools:
            observation = f"Ferramenta invalida: {action_name}. Ferramentas validas: {', '.join(sorted(tools))}."
        else:
            runtime = react_coala.ToolRuntime(
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

    return {
        "status": "incomplete",
        "final_answer": None,
        "steps": working_memory["step"],
        "tokens": total_tokens,
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "trajectory": trajectory,
        "working_memory": working_memory,
    }


def judge_attempt(question: str, attempt_result: dict[str, Any], llm: Any) -> dict[str, Any]:
    if not attempt_result.get("final_answer"):
        feedback = (
            "Faltou resposta final clara. Gere Final Answer assim que tiver todos os numeros necessarios "
            "e a comparacao final pedida pelo benchmark."
        )
        print("Judge Verdict: RETRY")
        print(f"Judge Feedback: {feedback}")
        return {
            "verdict": "RETRY",
            "feedback": feedback,
            "tokens": 0,
            "token_usage": zero_token_usage(),
            "llm_calls": 0,
        }

    reference = react_coala.get_benchmark_reference(question)
    if reference is not None:
        evaluation = react_coala.evaluate_benchmark_answer(
            question,
            attempt_result["final_answer"],
            reference=reference,
        )
        verdict = "ACCEPT" if evaluation["correct"] else "RETRY"
        print(f"Judge Verdict: {verdict}")
        print(f"Judge Feedback: {evaluation['feedback']}")
        return {
            "verdict": verdict,
            "feedback": evaluation["feedback"],
            "tokens": 0,
            "token_usage": zero_token_usage(),
            "llm_calls": 0,
        }

    response = llm.invoke(
        [
            SystemMessage(content="Voce avalia respostas de agentes com rigor e objetividade."),
            HumanMessage(
                content=build_reflection_judge_prompt(
                    question=question,
                    final_answer=attempt_result.get("final_answer"),
                    trajectory=attempt_result.get("trajectory", []),
                )
            ),
        ]
    )
    tokens = 0
    token_usage = zero_token_usage()
    if getattr(response, "usage_metadata", None):
        tokens += response.usage_metadata.get("total_tokens", 0)
        token_usage = add_token_usage(token_usage, response.usage_metadata)

    text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_judge_output(text)
    print(f"Judge Verdict: {parsed['verdict']}")
    print(f"Judge Feedback: {parsed['feedback']}")
    return {**parsed, "tokens": tokens, "token_usage": token_usage, "llm_calls": 1}


def reflect_on_attempt(question: str, attempt_result: dict[str, Any], feedback: str, llm: Any) -> dict[str, Any]:
    response = llm.invoke(
        [
            SystemMessage(content="Voce escreve reflexoes curtas e acionaveis para melhorar a proxima tentativa."),
            HumanMessage(
                content=build_reflection_prompt(
                    question=question,
                    attempt_result=attempt_result,
                    feedback=feedback,
                )
            ),
        ]
    )
    tokens = 0
    token_usage = zero_token_usage()
    if getattr(response, "usage_metadata", None):
        tokens += response.usage_metadata.get("total_tokens", 0)
        token_usage = add_token_usage(token_usage, response.usage_metadata)

    text = response.content if isinstance(response.content, str) else str(response.content)
    reflection = parse_reflection_output(text)
    print(f"Reflection: {reflection}")
    return {"reflection": reflection, "tokens": tokens, "token_usage": token_usage, "llm_calls": 1}


def run_reflection_agent(
    question: str,
    llm: Any,
    max_attempts: int = 3,
    max_steps: int = 6,
    memory_dir: str | Path = DEFAULT_REFLECTION_MEMORY_DIR,
) -> dict[str, Any]:
    root = Path(memory_dir)
    memory = react_coala.CoALAMemoryStore(root)
    reflection_memory = ReflectionMemoryStore(root)
    total_tokens = 0
    total_llm_calls = 0
    total_token_usage = zero_token_usage()
    last_attempt: dict[str, Any] | None = None
    last_feedback = ""
    started_at = time.perf_counter()

    print("=== INICIANDO AGENTE REFLECTION ===")
    print(f"Pergunta: {question}\n")
    print(f"Memoria persistente: {root.resolve()}\n")

    # Loop principal do Reflection: tentativa -> juiz -> reflexao -> nova tentativa.
    for attempt_number in range(1, max_attempts + 1):
        print(f"=== TENTATIVA {attempt_number}/{max_attempts} ===")
        reflections = reflection_memory.search_reflections(question, top_k=3)
        attempt_result = run_reflection_attempt(
            question=question,
            llm=llm,
            reflections=reflections,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            max_steps=max_steps,
            memory_dir=root,
        )
        total_tokens += attempt_result.get("tokens", 0)
        total_llm_calls += attempt_result.get("llm_calls", 0)
        total_token_usage = add_token_usage(total_token_usage, attempt_result.get("token_usage"))
        last_attempt = attempt_result

        judge_result = judge_attempt(question, attempt_result, llm)
        total_tokens += judge_result.get("tokens", 0)
        total_llm_calls += judge_result.get("llm_calls", 0)
        total_token_usage = add_token_usage(total_token_usage, judge_result.get("token_usage"))
        last_feedback = judge_result["feedback"]

        if judge_result["verdict"] == "ACCEPT":
            final_answer = attempt_result.get("final_answer") or "Sem resposta final."
            memory.add_episode(question, final_answer, react_coala._summarize_trajectory(attempt_result["trajectory"]))
            react_coala._auto_consolidate_semantic_memory(memory, question, final_answer)
            print("\n=== RESPOSTA FINAL ACEITA ===")
            return {
                "resposta": final_answer,
                "attempts": attempt_number,
                "steps": attempt_result["steps"],
                "tokens": total_tokens,
                "token_usage": total_token_usage,
                "llm_calls": total_llm_calls,
                "total_time_seconds": round(time.perf_counter() - started_at, 4),
                "trajectory": attempt_result["trajectory"],
                "judge_feedback": judge_result["feedback"],
                "memory_dir": str(root.resolve()),
                "memory_counts": {
                    **memory.counts(),
                    "reflections": reflection_memory.count(),
                },
            }

        if attempt_number < max_attempts:
            reflection_result = reflect_on_attempt(question, attempt_result, judge_result["feedback"], llm)
            total_tokens += reflection_result.get("tokens", 0)
            total_llm_calls += reflection_result.get("llm_calls", 0)
            total_token_usage = add_token_usage(total_token_usage, reflection_result.get("token_usage"))
            reflection_memory.add_reflection(
                question=question,
                reflection=reflection_result["reflection"],
                feedback=judge_result["feedback"],
                trajectory_summary=react_coala._summarize_trajectory(attempt_result["trajectory"]),
            )
            print("")

    final_answer = "Nao foi possivel obter uma resposta aceita dentro do limite de tentativas."
    memory.add_episode(
        question,
        final_answer,
        react_coala._summarize_trajectory(last_attempt["trajectory"] if last_attempt else []),
    )
    return {
        "resposta": final_answer,
        "attempts": max_attempts,
        "steps": last_attempt["steps"] if last_attempt else 0,
        "tokens": total_tokens,
        "token_usage": total_token_usage,
        "llm_calls": total_llm_calls,
        "total_time_seconds": round(time.perf_counter() - started_at, 4),
        "trajectory": last_attempt["trajectory"] if last_attempt else [],
        "judge_feedback": last_feedback,
        "memory_dir": str(root.resolve()),
        "memory_counts": {
            **memory.counts(),
            "reflections": reflection_memory.count(),
        },
    }
