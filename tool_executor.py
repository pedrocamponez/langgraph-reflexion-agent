import json
from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
"""
We need a ToolExecutor because we want to make this search async, instead of sync.
The ToolExecutor has a batch method in RunnableCallable->Runnable, which takes all the tool invocations and simply executes with a thread pool
running everything in parallel.
"""
tool_executor = ToolExecutor([tavily_tool])

"""
The execute_tools function is going to receive a STATE (which is basically a list of messages),
it will extract the tools that needs to be executed FROM those messages, it will EXECUTE them, and then it will
return a list of tool messages.
"""


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1] # the state[-1] means that we are always getting the last state when executing tools
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    # In tool_invocations, we are going to save the LangChain elements (objects with information of which tool (function) to use
    # and with which inputs to call it with.
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json",
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])
    outputs = tool_executor.batch(tool_invocations)

    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_))

    return tool_messages


if __name__ == '__main__':
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
                " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHichFFEmLitHFvFhKy1Ra",
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )

    print(raw_res)
