from dotenv import load_dotenv

import os


load_dotenv()

if "GROQ_API_KEY" not in os.environ or "LANGSMITH_API_KEY" not in os.environ:
    raise SystemExit(0)

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI



class Source(BaseModel):
    """Schema for a source used by the agent"""

    urd:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer:str = Field(description="Answer to the agent's query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to get the answer")


llm = ChatGroq(model="openai/gpt-oss-120b")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.1-pro-preview",
# )
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-search!")
    response = agent.invoke({"messages": HumanMessage(content="Search for 3 job posting for an Fullstack developer on linkedin")})
    print(response)

if __name__ == "__main__":
    main()
