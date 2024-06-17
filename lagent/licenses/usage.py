import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)

prompt_template = """
Here is a software licence text : 

===== licence text =====

{licence_text}

===== end of licence text =====

You are a jurist expert in software licensing. 

You have been asked to rewrite this full license text as a list of formal end-user usage cases, each written with the exact format:
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

Answer the user query. Wrap the output in `json` tags. Output only the list of usage cases.

{format_instructions}

"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class UsageCase(BaseModel):
    title: str = Field(description="the title of the usage case")
    description: str = Field(description="the description of the usage case")
    allowed: bool = Field(description="the statement that the usage case is allowed with or without conditions (true) or not allowed (false)")
    conditions: str = Field(description="if any, the conditions for the usage case to be allowed.")

class UsageCases(BaseModel):
    usage_cases: List[UsageCase]


parser = PydanticOutputParser(pydantic_object=UsageCases)
prompt = ChatPromptTemplate.from_template(prompt_template).partial(format_instructions=parser.get_format_instructions())
