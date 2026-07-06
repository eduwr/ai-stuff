import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
 
load_dotenv()

if __name__ == "__main__":
    print("Hello from 08-reflexion!")
