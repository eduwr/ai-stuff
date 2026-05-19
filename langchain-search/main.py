from dotenv import load_dotenv

import getpass
import os


load_dotenv()

if "GROQ_API_KEY" not in os.environ or "LANGSMITH_API_KEY" not in os.environ:
    raise SystemExit(0)

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


llm = ChatGroq(model="qwen/qwen3-32b")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-search!")
    response = agent.invoke({"messages": HumanMessage(content="What's the weather in Tokyo?")})
    print(response)

if __name__ == "__main__":
    main()
