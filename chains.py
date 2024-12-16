import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion

llm = ChatOpenAI(model="gpt-4-turbo-preview")
parser = JsonOutputToolsParser(return_id=True)
# The parser_pydantic is going to take the answer from the LLM and transform it into an AnswerQuestion object
# so we can work with it.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

"""
Use this technique to GROUND the LLM to use the Pydantic AnswerQuestion object we created, so it won't diverge or
use its own knowledge to answer. Tools are "functions" the LLM can call to produce outputs in a structured 
and predictable format. They are particularly helpful for:
1. Grounding the model to specific actions or outputs;
2. Interfacing the LLM with external systems;
3. Constraining the model's response to ensure they follow predefined formats or schemas.
In this case, this is the tool:
```
class AnswerQuestion(BaseModel):
    Answer the question.

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
``` 

This tool specifies that the output must be structured as a JSON object, and contains the fields.

Here in the first_responder, llm.bind_tools() is used to attach the AnswerQuestion tool to the LLM call, so we are
constraining the LLM to only produce outputs that match the AnswerQuestion schema we defined earlier.
The ```tool_choice="AnswerQuestion"``` explicitly tells the LLM which tool it must use for generating the output.
"""
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
