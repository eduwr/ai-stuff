from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client

llm = ChatOpenAI(temperature=0)
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

generation_chain = prompt | llm | StrOutputParser()

