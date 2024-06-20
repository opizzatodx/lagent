import logging
import os
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


class Agent():

    def __init__(self, licenses_data={}):
        self.licenses_data = licenses_data

        # agent state
        self.user_license_tag = None
        self.user_license_name = None
        self.user_license_use_cases = None
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

        # get the license list
        self.license_names = [item["name"] for item in self.licenses_data.values()]

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
        self.user_license_tag = None
        self.user_license_name = None
        self.user_license_use_cases = None
        self.user_use_case = None            
        logger.info("chat history is cleared")

    def chat(self, message, history):

        if len(history) == 0:
            self.clear_history()

        if self.user_license_tag is None:
            return self.chat_for_matching_the_license(message)

        if self.user_use_case is None:
            return self.chat_for_matching_the_use_case(message)

        return self.chat_for_response(message)

    def clear_history_from_last_ai_result_and_pop_last_human_input(self):
            
        # remove the last message, supposed to be the AI message
        self.chat_history.messages = self.chat_history.messages[:-1]

        # pop the next last message, supposed to be the last human input
        initial_human_input = self.chat_history.messages[-1].content
        self.chat_history.messages = self.chat_history.messages[:-1]

        return initial_human_input

    def chat_for_matching_the_license(self, user_input):

        logger.info(f"entering LICENSE chat")
        try:
            res = self.license_chain.invoke(
                {"input": user_input, "license_database": ", ".join(self.license_names)},
                {"configurable": {"session_id": "unused"}},
            )
        except Exception as e:
            logger.error(f"error during license_chain_with_message_history.invoke, {e=}")
            return f"TECHNICAL ERROR: {e}"

        logger.info(f"LICENSE chain response: {res}")

        # find the line with the final answer from the LLM response
        final_answer_line = find_a_line_starting_with(res.split("\n"), "Final Answer:")
        if final_answer_line is None:
            # no final answer found, return the response as is
            return res

        # extract the license name from the LLM response
        license_name = final_answer_line.replace("Final Answer: ", "").strip()
        logger.info(f"License {license_name=} in final answer")

        # get the license 
        matching_licenses = [item for item in self.licenses_data.values() if item["name"] == license_name]
        if len(matching_licenses) == 0:
            logger.error(f"{license_name=} not found in the database. {self.license_names=}")
            return res

        # update the agent state with the license and its use cases
        matching_license = matching_licenses[0]
        self.user_license_tag = matching_license["tag"]
        self.user_license_name = license_name
        self.user_license_use_cases = self.licenses_data[self.user_license_tag]["use_cases"]
        logger.info(f"License {license_name} found in the database: {self.user_license_tag=}, {len(self.user_license_use_cases)=}")

        # reset history to previous conversation and call the next chat
        human_input = self.clear_history_from_last_ai_result_and_pop_last_human_input()
        res = self.chat_for_matching_the_use_case(human_input)
        return res

    def chat_for_matching_the_use_case(self, user_input):

        logger.info(f"entering USE CASE chat for license {self.user_license_tag}")

        response_prefix = f"""
        I understand you are asking about the license: {self.user_license_name}.
        """

        cases = [item["description"] for item in self.user_license_use_cases]

        res = self.use_case_chain.invoke(
            {"input": user_input, "use_cases_database": " | ".join(cases)},
            {"configurable": {"session_id": "unused"}},
        )
        logger.info(f"USE CASE chain response: {res}")

        # find the line with the final answer from the LLM response
        final_answer_line = find_a_line_starting_with(res.split("\n"), "Final Answer:")
        if final_answer_line is None:
            # no final answer found, return the response as is
            return response_prefix + res

        use_case = final_answer_line.replace("Final Answer: ", "").strip()

        # get the use case
        matching_cases = [item for item in self.user_license_use_cases if item["description"] in use_case]
        if len(matching_cases) == 0:
            logger.error(f"Use case {use_case} not found in the database.")
            logger.info("use cases available for this license:")
            for c in self.user_license_use_cases:
                logger.info(f"{c['description']}")
            res = f"TECHNICAL ERROR: Use case '{use_case}' not found in the license database."
            return response_prefix + res

        # update the agent state with the use case
        matching_case = matching_cases[0]
        self.user_use_case = matching_case
        logger.info(f"Use case {use_case} found in the database")

        # reset history to previous conversation and call the next chat
        human_input = self.clear_history_from_last_ai_result_and_pop_last_human_input()
        res = self.chat_for_response(human_input)
        return res

    def chat_for_response(self, user_input):

        logger.info(f"entering RESPONSE chat for {self.user_license_tag=} and {self.user_use_case=}")

        response_prefix = f"""
        I understand you are asking about the license: {self.user_license_name}.
        I understand your use case is related to: {self.user_use_case['description']}
        """

        res = self.response_chain.invoke(
            {"input": user_input, 
             "license_use_case": f"{self.user_use_case}",
             "license": f"{self.user_license_name}",
             },
            {"configurable": {"session_id": ""}},
        )
        logger.info(f"RESPONSE chain response: {res}")

        return response_prefix + res

