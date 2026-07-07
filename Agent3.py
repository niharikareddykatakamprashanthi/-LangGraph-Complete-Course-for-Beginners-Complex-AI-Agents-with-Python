# ReACT Agent
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int) -> int:
    """Add two numbers."""
    return a + b

tools = [add]

model = ChatOpenAI(model_name="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
                                  "You are my AI assistant, please answer my query to the best of your ability."
                                  )
    response = model.invoke([system_prompt]+ state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1] 
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)
tool_node = ToolNode(tools)
graph.add_node("tools",tool_node)
graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent", 
    should_continue, 
    {
        "continue": "tools",
        "end": END
    },
    )
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_output(stream):
    printed = 0
    for s in stream:
        messages = s["messages"]
        for message in messages[printed:]:
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
        printed = len(messages)

input = AgentState(messages=[("user","Add 34 + 21. Add 10 and 12")])
print_output(app.stream(input, stream_mode="values"))

