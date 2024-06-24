import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


logger = logging.getLogger(__name__)

prompt_template = """
Here is a software licence text : 

===== licence text =====

{licence_text}

===== end of licence text =====

You are a jurist expert in software licensing. 

You have been asked to rewrite this full license text as a list of formal end-user use cases, each written with the exact format:
- a title
- a description of the case
- the statement that this case is allowed or not allowed or allowed with conditions
- conditions if any must be in a one line full text without bullet points or list format. No double quotes or special characters are allowed in the conditions text.

This list should cover the full original text.

This list must include all the following standard cases:
- the right to use the software for personal use
- the right to use the software for commercial use
- the right to modify the software
- the right to distribute the software
- the right to sublicense the software

The list must include all the use cases that are allowed and all the use cases that are not allowed in the original text.

Answer the user query. Wrap the output in `json` tags. Output only the list of use cases.

{format_instructions}

"""

class UseCase(BaseModel):
    title: str = Field(description="the title of the use case")
    description: str = Field(description="the description of the use case")
    allowed: bool = Field(description="the statement that the use case is allowed with or without conditions (true) or not allowed (false)")
    conditions: str = Field(description="if any, the conditions for the use case to be allowed.")

class UseCases(BaseModel):
    use_cases: List[UseCase]


parser = PydanticOutputParser(pydantic_object=UseCases)
prompt = ChatPromptTemplate.from_template(prompt_template).partial(format_instructions=parser.get_format_instructions())
