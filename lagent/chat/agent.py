import logging
import os
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails import RailsConfig

from lagent.common.utils import find_a_line_starting_with
from .prompt.license import prompt as license_prompt
from .prompt.use_case import prompt as use_case_prompt
from .prompt.response import prompt as response_prompt

# https://github.com/langchain-ai/langchain/discussions/21596
logging.getLogger("langchain_core.tracers.base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Represent the state of the agent during the conversation.
    There are three stages in the conversation:
    - match_the_license: identify the license from the user input
    - match_the_use_case: identify the use case from the user input
    - give_answer: generate the response to the user question
    """

    # variables when the license is identified
    user_license_tag: str
    user_license_name: str
    user_license_use_cases: List[str]

    # variables when the use case is identified
    user_use_case: str

    # list of messages exchanged with the user
    messages: Annotated[list, add_messages]


# LangGraph function that determines if the agent should move to the stage "match_the_use_case"
def should_match_the_use_case(state: AgentState) -> Literal["match_the_use_case", END]:
    if state.get("user_license_tag") is not None:
        return "match_the_use_case"
    return END

# LangGraph function that determines if the agent should move to the stage "give_answer"
def should_give_answer(state: AgentState) -> Literal["give_answer", END]:
    if state.get("user_use_case") is not None:
        return "give_answer"
    return END


class Agent:

    def __init__(self, licenses_data={}, graph_draw_path=None):

        self.licenses_data = licenses_data

        # part of the agent state is duplicated here in order to be accessed 
        # by the guardrails prompt context callback during the response stage
        self.user_license_name = None
        self.user_use_case = None

        # LLM model
        nvapi_key = os.environ["NVIDIA_API_KEY"]
        self.llm = ChatNVIDIA(model="mistralai/mistral-large", nvidia_api_key=nvapi_key, max_tokens=2048, temperature=0, stop=["User", "input", ":"])

        self.output_parser = StrOutputParser()
        self.chat_history = ChatMessageHistory()

        # chain for the license and use_case stages
        self.license_chain = self.build_chain_from_prompt(license_prompt)
        self.use_case_chain = self.build_chain_from_prompt(use_case_prompt)

        # chain for the response stage, with guardrails for verifying consistency of llm answer with license and use case
        config = RailsConfig.from_path("./guardrails/config")
        guardrails = RunnableRails(config)
        guardrails.rails.register_prompt_context("license_name", lambda: self.user_license_name)
        guardrails.rails.register_prompt_context("use_case", lambda: self.user_use_case)
        self.response_chain = self.build_chain_from_prompt(response_prompt, guardrails=guardrails)

        # define the LangGraph workflow
        workflow = StateGraph(AgentState)

        # a node for each stage of the conversation
        workflow.add_node("match_the_license", self.call_license_chain)
        workflow.add_node("match_the_use_case", self.call_use_case_chain)
        workflow.add_node("give_answer", self.call_give_answer_chain)

        # entry point and edges with conditional functions
        workflow.set_entry_point("match_the_license")
        workflow.add_conditional_edges("match_the_license", should_match_the_use_case)
        workflow.add_conditional_edges("match_the_use_case", should_give_answer)
        workflow.add_edge("give_answer", END)
        self.workflow = workflow
        self.graph_chain = None

        # build the LangChain runnable chain
        self.graph_chain = self.workflow.compile(checkpointer=MemorySaver())

        # draw the graph and print it
        self.graph_chain.get_graph().print_ascii()
        if graph_draw_path is not None:
            self.graph_chain.get_graph().draw_mermaid_png(output_file_path=graph_draw_path)

    # main chat function that handles the conversation stages
    def chat(self, message, history):

        if len(history) == 0:
            self.chat_history.clear()
            logger.info("chat history is cleared")

        res = self.graph_chain.invoke(
            {"messages": [HumanMessage(content=message)]},
            {"configurable": {"session_id": "unused", "thread_id": "unused"}},
        )
        res = res["messages"][-1].content
        return res

    # function to call the license chain
    def call_license_chain(self, state: AgentState):

        logger.info(f"entering LICENSE chat")

        license_names = [item["name"] for item in self.licenses_data.values()]

        user_input = state["messages"][-1]
        try:
            res = self.license_chain.invoke(
                {"input": user_input, "license_database": ", ".join(license_names)},
                {"configurable": {"session_id": "unused"}},
            )
        except Exception as e:
            logger.error(f"error during license_chain_with_message_history.invoke, {e=}")
            res = f"TECHNICAL ERROR: {e}"
            return {
                "messages": [res],
            }

        logger.info(f"LICENSE chain response: {res}")

        # find the line with the final answer from the LLM response
        final_answer_line = find_a_line_starting_with(res.split("\n"), "Final Answer:")
        if final_answer_line is None:
            # no final answer found, return the response as is
            return {
                "messages": [res],
            }

        # extract the license name from the LLM response
        license_name = final_answer_line.replace("Final Answer: ", "").strip()
        logger.info(f"License {license_name=} in final answer")

        # get the license 
        matching_licenses = [item for item in self.licenses_data.values() if item["name"] == license_name]
        if len(matching_licenses) == 0:
            logger.error(f"{license_name=} not found in the database. {license_names=}")
            return {
                "messages": [res],
            }

        self.user_license_name = license_name

        # retrieve the license use cases
        user_license_tag = matching_licenses[0]["tag"]
        user_license_use_cases = self.licenses_data[user_license_tag]["use_cases"]
        logger.info(f"License {license_name} found in the database: {user_license_tag=}, {len(user_license_use_cases)=}")

        # return an updated state with the license information
        return {
            "messages": [res], 
            "user_license_tag": user_license_tag,
            "user_license_name": license_name,
            "user_license_use_cases": user_license_use_cases,
        }

    # function to call the use case chain
    def call_use_case_chain(self, state: AgentState):

        logger.info(f"entering USE CASE chat for license {state['user_license_tag']=}")

        response_prefix = f"""
        I understand you are asking about the license: {state['user_license_name']}.
        """

        cases = [item["description"] for item in state['user_license_use_cases']]

        user_input = state["messages"][-1]
        res = self.use_case_chain.invoke(
            {"input": user_input, "use_cases_database": " | ".join(cases)},
            {"configurable": {"session_id": "unused"}},
        )
        logger.info(f"USE CASE chain response: {res}")

        # find the line with the final answer from the LLM response
        final_answer_line = find_a_line_starting_with(res.split("\n"), "Final Answer:")
        if final_answer_line is None:
            # no final answer found, return the response as is
            return {**state, **{
                "messages": [response_prefix + res],
            }}

        use_case = final_answer_line.replace("Final Answer: ", "").strip()

        # get the use case
        matching_cases = [item for item in state['user_license_use_cases'] if item["description"] in use_case]
        if len(matching_cases) == 0:
            logger.error(f"Use case {use_case} not found in the database.")
            logger.info("use cases available for this license:")
            for c in self.user_license_use_cases:
                logger.info(f"{c['description']}")
            res = f"TECHNICAL ERROR: Use case '{use_case}' not found in the license database."
            return {**state, **{
                "messages": [response_prefix + res],
            }}

        # update the agent state with the use case
        use_case = matching_cases[0]
        self.user_use_case = use_case
        logger.info(f"Use case {use_case} found in the database")

        return {**state, **{
            "messages": [response_prefix + res],
            "user_use_case": use_case,
            }
        }

    # function to call the response chain
    def call_give_answer_chain(self, state: AgentState):

        print("call_give_answer_chain")

        logger.info(f"entering RESPONSE chat for {state['user_license_tag']=} and {state['user_use_case']=}")

        response_prefix = f"""
        I understand you are asking about the license: {state['user_license_name']}.
        I understand your use case is related to: {state['user_use_case']['description']}
        """

        user_input = state["messages"][-1]
        res = self.response_chain.invoke(
            {"input": user_input, 
             "license_use_case": f"{state['user_use_case']}",
             "license": f"{state['user_license_name']}",
             },
            {"configurable": {"session_id": ""}},
        )
        logger.info(f"RESPONSE chain response: {res}")

        return {**state, **{
            "messages": [response_prefix + res], 
        }}


    def build_chain_from_prompt(self, prompt, guardrails=None):

        if guardrails is not None:
            chain = prompt | (guardrails | self.llm) | self.output_parser
        else:
            chain = prompt | self.llm | self.output_parser

        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return chain_with_message_history

    def clear_history(self):

        self.chat_history.clear()
        self.user_license_name = None
        self.user_use_case = None

        # rebuild the graph chain to clear the state and history
        self.graph_chain = self.workflow.compile(checkpointer=MemorySaver())

        logger.info("chat history is cleared")
