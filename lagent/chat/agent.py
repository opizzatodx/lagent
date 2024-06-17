import logging
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from lagent.common.utils import find_a_line_starting_with
from .license import prompt as license_prompt
from .usage import prompt as usage_prompt
from .response import prompt as response_prompt

logger = logging.getLogger(__name__)


class LicenseAgent():

    def __init__(self, licenses_data={}):
        self.licenses_data = licenses_data

        # agent state
        self.user_license_tag = None
        self.user_license_name = None
        self.user_license_usage_cases = None
        self.user_usage_case = None

        # LLM model
        nvapi_key = os.environ["NVIDIA_API_KEY"]
        self.llm = ChatNVIDIA(model="mistralai/mistral-large", nvidia_api_key=nvapi_key, max_tokens=2048, temperature=0, stop=["User", "input", ":"])

        self.output_parser = StrOutputParser()
        self.chat_history = ChatMessageHistory()

        # build chain for license matching
        license_chain = license_prompt | self.llm | self.output_parser
        self.license_chain_with_message_history = RunnableWithMessageHistory(
            license_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # build chain for usage matching
        usage_chain = usage_prompt | self.llm | self.output_parser
        self.usage_chain_with_message_history = RunnableWithMessageHistory(
            usage_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # build chain for usage matching
        response_chain = response_prompt | self.llm | self.output_parser
        self.response_chain_with_message_history = RunnableWithMessageHistory(
            response_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # get the license list
        self.license_names = [item["name"] for item in self.licenses_data.values()]

    def clear_history(self):
        self.chat_history.clear()
        self.user_license_tag = None
        self.user_license_name = None
        self.user_license_usage_cases = None
        self.user_usage_case = None            
        logger.info("chat history is cleared")

    def chat(self, message, history):

        if len(history) == 0:
            self.clear_history()

        if self.user_license_tag is None:
            return self.chat_for_matching_the_license(message)

        if self.user_usage_case is None:
            return self.chat_for_matching_the_use_case(message)

        return self.chat_for_response(message)

    def chat_for_matching_the_license(self, message):

        logger.info(f"entering LICENSE chat")
        user_input = message

        try:
            res = self.license_chain_with_message_history.invoke(
                {"input": user_input, "license_database": ", ".join(self.license_names)},
                {"configurable": {"session_id": "unused"}},
            )
        except Exception as e:
            logger.error(f"error during license_chain_with_message_history.invoke, {e=}")
            return f"TECHNICAL ERROR: {e}"

        logger.info(f"license chain response: {res}")

        final_answer_line = find_a_line_starting_with(res.split("\n"), "Final Answer:")
        if final_answer_line is None:
            return res

        # license name found
        license_name = final_answer_line.replace("Final Answer: ", "").strip()
        logger.info(f"License {license_name=} in final answer")

        # get the license 
        matching_licenses = [item for item in self.licenses_data.values() if item["name"] == license_name]
        if len(matching_licenses) == 0:
            logger.error(f"{license_name=} not found in the database. {self.license_names=}")
            return res

        matching_license = matching_licenses[0]
        self.user_license_tag = matching_license["tag"]
        self.user_license_name = license_name
        logger.info(f"License {license_name} found in the database with tag: {self.user_license_tag}")

        # get usage cases for the license
        cases = self.licenses_data[self.user_license_tag]["usage_cases"]
        logger.info(f"found {len(cases)} usage cases for license {license_name} with tag {self.user_license_tag}")
        self.user_license_usage_cases = cases

        # remove last AI message which is the result of use case matching
        # and get the last human message to use as human input
        previous_history = self.chat_history.messages
        previous_history = previous_history[:-1]
        initial_human_input = previous_history[-1].content
        previous_history = previous_history[:-1]
        self.chat_history.messages = previous_history

        res = self.chat_for_matching_the_use_case(initial_human_input)
        return res

    def chat_for_matching_the_use_case(self, message):

        logger.info(f"entering USAGE chat for license {self.user_license_tag}")

        user_input = message
        reponse_prefix = f"""
        I understand you are asking about the license: {self.user_license_name}.
        """

        cases = [item["description"] for item in self.user_license_usage_cases]
        res = self.usage_chain_with_message_history.invoke(
            {"input": user_input, "usages_database": " | ".join(cases)},
            {"configurable": {"session_id": "unused"}},
        )
        logger.info(f"usage chain response: {res}")

        last_answer = res.split("\n")[-1].strip()
        if "Final Answer:" not in last_answer:
            return reponse_prefix + res

        usage_case = last_answer.replace("Final Answer: ", "").strip()

        # get the usage case
        matching_cases = [item for item in self.user_license_usage_cases if item["description"] in usage_case]
        if len(matching_cases) == 0:
            logger.error(f"Usage case {usage_case} not found in the database.")
            print("cases:")
            for c in self.user_license_usage_cases:
                print(c)
            res = f"TECHNICAL ERROR: Usage case {usage_case=} not found in the database."
            return reponse_prefix + res

        matching_case = matching_cases[0]
        logger.info(f"Usage case {usage_case} found in the database")
        self.user_usage_case = matching_case

        # remove last AI message which is the result of use case matching
        # and get the last human message to use as human input
        previous_history = self.chat_history.messages
        previous_history = previous_history[:-1]
        initial_human_input = previous_history[-1].content
        previous_history = previous_history[:-1]
        self.chat_history.messages = previous_history

        res = self.chat_for_response(initial_human_input)
        return res

    def chat_for_response(self, message):

        logger.info(f"entering RESPONSE chat for license {self.user_license_tag=} and usage case {self.user_usage_case=}")

        user_input = message

        reponse_prefix = f"""
        I understand you are asking about the license: {self.user_license_name}.
        I understand your usage case is related to: {self.user_usage_case['description']}
        """

        res = self.response_chain_with_message_history.invoke(
            {"input": user_input, "license_usage_case": f"{self.user_usage_case}"},
            {"configurable": {"session_id": ""}},
        )
        logger.info(f"response chain response: {res}")

        return reponse_prefix + res

