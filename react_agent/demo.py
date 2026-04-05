import os
import openai
from dotenv import load_dotenv

from agent import ReActAgent
from schemas import AgentConfig

load_dotenv()


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    client = openai.OpenAI(api_key=api_key)
    agent = ReActAgent(client, config=AgentConfig(max_steps=15, max_verify_attempts=2, temperature=0.7))

    question = "What is the time complexity of Python's built-in sort (Timsort) in the worst case? If I sort a list of 1 million elements, and each comparison takes 10 nanoseconds, what is the maximum time in milliseconds the sort could take?"
    print(f"Question: {question}\n")

    result = agent.run(question)

    print("--- Final Answer ---")
    print(result.answer)
    print(f"\nVerified: {result.verified}")
    print("\n--- Agent Trace ---")
    for entry in result.trace:
        print(entry)
