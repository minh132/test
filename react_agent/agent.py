import json
import openai
from typing import Any, Dict, List, Tuple

from tools.registry import TOOL_SCHEMAS, TOOL_DISPATCH
from schemas import AgentConfig, AgentRun


def _retry_on_incorrect(fn):
    """Re-runs fn up to self.config.max_verify_attempts times on INCORRECT verdicts."""
    def wrapper(self, question: str, messages: List[Dict]) -> Tuple[str, bool]:
        for attempt in range(self.config.max_verify_attempts):
            answer, is_correct, feedback = fn(self, question, messages)
            if is_correct:
                return answer, True
            self.trace.append(f"\n--- Retry {attempt + 1} after verification failure ---")
            messages.append({
                "role": "user",
                "content": (
                    "The verification found an issue with your answer. "
                    "Use tools again if needed to correct it. "
                    f"Feedback:\n{feedback}\nProvide a corrected final answer."
                ),
            })
        return answer, False
    return wrapper


class ReActAgent:
    MODEL = "gpt-4o-mini"

    def __init__(self, llm_client: openai.OpenAI, config: AgentConfig = None):
        self.client = llm_client
        self.config = config or AgentConfig()
        self.trace: List[str] = []

        self.system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "Use tools whenever you need external information or to perform computations. "
            "Think step-by-step before deciding whether to call a tool. "
            "Once you have enough information to fully address the user's question, "
            "respond with a clear, concise final answer in plain text — do not call any more tools."
        )

    def _chat(self, messages: List[Dict[str, Any]], tools: Any = None, temperature: float = None):
        kwargs: Dict[str, Any] = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs).choices[0].message

    def _call_llm_text(self, messages: List[Dict[str, Any]]) -> str:
        return self._chat(messages, temperature=0.0).content or ""

    def _self_verify(self, question: str, proposed_answer: str) -> Tuple[bool, str]:
        trace_text = "\n".join(self.trace)
        verify_prompt = f"""Question: {question}

Reasoning trace and tool observations:
{trace_text}

Proposed Answer: {proposed_answer}

Verify the Proposed Answer against the trace above:
- Does it directly address the original question?
- Is it consistent with every tool observation in the trace?
- Are there any logical errors, contradictions, or unsupported claims?

Trust tool observations as ground truth.

Output exactly 'CORRECT' if the answer is sound.
Otherwise output 'INCORRECT' followed by a brief explanation."""

        feedback = self._call_llm_text([
            {
                "role": "system",
                "content": "You are a strict verification assistant. Verify the answer based on the observations in the trace.",
            },
            {"role": "user", "content": verify_prompt},
        ])
        return feedback.strip().upper().startswith("CORRECT"), feedback

    @_retry_on_incorrect
    def _run_until_answer(self, question: str, messages: List[Dict]) -> Tuple[str, bool, str]:
        for _ in range(self.config.max_steps):
            response_msg = self._chat(messages, tools=TOOL_SCHEMAS)

            if response_msg.tool_calls:
                messages.append(response_msg)
                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    self.trace.append(f"Action: {fn_name}\nAction Input: {json.dumps(fn_args)}")
                    result = TOOL_DISPATCH[fn_name](fn_args) if fn_name in TOOL_DISPATCH else f"Error: Unknown tool '{fn_name}'"
                    self.trace.append(f"Observation: {result}")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
                continue

            final_answer = response_msg.content or ""
            self.trace.append(f"Final Answer: {final_answer}")
            messages.append({"role": "assistant", "content": final_answer})

            self.trace.append("--- Self-Verification ---")
            is_correct, feedback = self._self_verify(question, final_answer)
            self.trace.append(f"Verification: {feedback}")
            return final_answer, is_correct, feedback

        return "Failed to produce an answer within the step limit.", False, ""

    def run(self, question: str) -> AgentRun:
        self.trace = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        answer, verified = self._run_until_answer(question, messages)
        return AgentRun(question=question, answer=answer, trace=list(self.trace), verified=verified)
