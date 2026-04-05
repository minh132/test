import os
import openai
from dotenv import load_dotenv

from agent import ReActAgent

load_dotenv()


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    client = openai.OpenAI(api_key=api_key)
    agent = ReActAgent(client)

    answer = agent.run("What is 12% of the GDP of France in 2023 (in USD)?")

    print("--- Final Answer ---")
    print(answer)
    print("\n--- Agent Trace ---")
    for entry in agent.trace:
        print(entry)
