import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from chains import llm, reflect_chain, generate_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END,REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello from 07-reflexion!")

    inputs = HumanMessage(content="""Make this tweet better: "
                            Folks, I'm going to try to explain something serious.

                            An LLM is, essentially, a sausage-making machine.

                            The sausage-making machine takes, on one side, pieces of meat of dubious origin and, on the other side, out comes a nice strip of sausages.

                            If you shove the sausages in backwards, you can't get the dubious-origin meat back. No matter what you put in, only more sausage will come out.

                            I hope that helped.
                          
                          "
                          
                          """)
    
    response = graph.invoke(inputs)

    print(response)