# Exercise 2 — Tool-Calling Agent with Self-Verification

A ReAct-style agent using OpenAI native function calling with two tools and a self-verification step.

## Architecture

| Component | Description |
|---|---|
| `ReActAgent` | Main agent loop — calls LLM with tool schemas, dispatches tools, feeds results back, then self-verifies. |
| `AgentConfig` | Dataclass for configuring `max_steps`, `max_verify_attempts`, and `temperature`. |
| `AgentRun` | Return type: contains `answer`, `trace`, and `verified` flag. |
| `calculator` | Safely evaluates math expressions using `eval` with restricted globals. |
| `web_search` | Searches the web via the Tavily API and returns combined result snippets. |

## Agent Loop

```
User question
     │
     ▼
 LLM call (with tool schemas)
     │
     ├── tool_calls? ──► dispatch tool ──► feed result back ──► LLM call again
     │
     └── plain text ──► final answer
                              │
                              ▼
                    Self-verification LLM call
                              │
                    ┌─────────┴─────────┐
                  CORRECT           INCORRECT
                    │                   │
                 return          retry via decorator
```

## Retry Strategy

The agent uses a `@_retry_on_incorrect(max_attempts=N)` decorator instead of a while-loop counter. Each failed verification injects correction feedback into the message history before the next attempt. After `max_verify_attempts` failures the best-effort answer is returned.

## Self-Verification

After producing a final answer, the agent issues one more LLM call with the full reasoning trace. The verifier checks:
- Does the answer directly address the question?
- Is it consistent with every tool observation?
- Are there any logical errors or contradictions?

## Configuring the agent

```python
from agent import ReActAgent
from schemas import AgentConfig

config = AgentConfig(
    max_steps=20,
    max_verify_attempts=3,
    temperature=0.5,
)
agent = ReActAgent(client, config=config)
result = agent.run("What is 12% of France's 2023 GDP in USD?")
print(result.answer)
print(result.verified)
```

## Safe Calculator

`eval` is run with `__builtins__` stripped and only `abs` / `round` whitelisted:

```python
allowed_names = {"abs": abs, "round": round}
code = compile(expression, "<string>", "eval")
for name in code.co_names:
    if name not in allowed_names:
        raise NameError(f"Use of {name} not allowed")
result = eval(code, {"__builtins__": {}}, allowed_names)
```

## Setup

```bash
pip install openai python-dotenv
```

Create a `.env` file in `react_agent/`:

```
OPENAI_API_KEY=sk-...
```

## Run the demo

```bash
cd react_agent/
python demo.py
```

## Sample trace

```
Action: web_search
Action Input: {"query": "France GDP 2023 in USD"}
Observation: France's GDP in 2023 was approximately $2.78 trillion USD (World Bank estimate).
Action: calculator
Action Input: {"expression": "0.12 * 2.78e12"}
Observation: 333600000000.0
Final Answer: 12% of the GDP of France in 2023 is approximately $333.6 billion USD.
--- Self-Verification ---
Verification: CORRECT
```
