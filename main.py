from dotenv import load_dotenv

from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2
builder = MessageGraph()
# Name of the nome is "Draft", and the executor is going to generate the first graph, built-in critique and search tools
builder.add_node("draft", first_responder)
# Execute_tools node will run the execute_tools function, which takes in the input of the state, run the tavily search
# queries and return real time data
builder.add_node("execute_tools", execute_tools)
# Revisor chain
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    """This function is going to run after the revisor node, and decide what to do next:
    are we going to end, or run it again?"""
    count_tool_visists = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visists
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == '__main__':
    print("Hello Reflexion")
    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous SOC problem domain, list startups that do that and raised capital."
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
