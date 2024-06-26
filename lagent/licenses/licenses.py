import logging
import os
from typing import List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field, ValidationError
from langchain_core.pydantic_v1 import ValidationError as V1ValidationError

from lagent.common.utils import load_text_file_lines, read_yaml_file, write_yaml_file
from lagent.licenses.prompt.use_case import prompt, parser, UseCases

logger = logging.getLogger(__name__)


class LicenseAndLink(BaseModel):
    tag: str = Field(description="the title of the use case")
    name: str = Field(description="the description of the use case")
    link: str = Field(description="the statement that the use case is allowed with or without conditions (true) or not allowed (false)")

class LicensesAndLink(BaseModel):
    licenses_and_link: List[LicenseAndLink]


class LicenseProcessor():

    def __init__(self, licenses_and_link_file_path, licenses_text_dir, licenses_use_cases_dir):

        self.licenses_and_link_file_path = licenses_and_link_file_path
        self.licenses_text_dir = licenses_text_dir
        self.licenses_use_cases_dir = licenses_use_cases_dir

        # LLM model
        nvapi_key = os.environ["NVIDIA_API_KEY"]
        self.llm = ChatNVIDIA(model="mistralai/mistral-large", nvidia_api_key=nvapi_key, max_tokens=2048, temperature=0, stop=["User", "input", ":"])

        self.chain = prompt | self.llm | parser

    def read_licenses_and_link(self):
    
        data = read_yaml_file(file_path=self.licenses_and_link_file_path)
        if (data is None):
            logger.error(f"error reading licenses file {self.licenses_and_link_file_path=}")
            return None

        logger.info(f"read licenses file {self.licenses_and_link_file_path=}, {len(data)=}")

        try:
            licenses = LicensesAndLink.model_validate(data).model_dump()["licenses_and_link"]
        except ValidationError as error:
            logger.error(f"error parsing licenses file content, {error=}")
            return None

        return licenses
    
    def read_license_text(self, license_tag):
        file_path = os.path.join(self.licenses_text_dir, license_tag + ".txt")
        data = load_text_file_lines(file_path)
        if (data is None):
            logger.error(f"error reading license text for {license_tag=}")
            return None
        data_str = "".join(data)
        logger.info(f"read license file {file_path=}, {len(data_str)=}")    
        return data

    def generate_license_use_cases(self, license_tag):

        license_content = self.read_license_text(license_tag)
        license_str = "".join(license_content)
        logger.info(f"processing license: {license_tag}, {len(license_str)=}")

        try:
            result = self.chain.invoke({"licence_text": license_str})
        except Exception as e:
            logger.error(f"error processing license: {license_tag}, {e=}")
            return None

        logger.info(f"{len(result.use_cases)=}")
        for uc in result.use_cases:
            logger.info(f"{uc.title=}")

        res = result.dict()
        return res

    def read_license_use_cases(self, license_tag):

        file_path = os.path.join(self.licenses_use_cases_dir, license_tag + ".yaml")

        log_prefix = f"read use cases file ${file_path}, "

        data = read_yaml_file(file_path=file_path, skip_root_tag="use_cases")
        if (data is None):
            logger.error(log_prefix + f"error reading file")
            return None
        logger.info(log_prefix + f"{len(data)=}")

        try:
            # pydantic_v1 validation (because this object is defined for LangChain PydanticOutputParser)
            cases = UseCases.validate({"use_cases": data}).dict()["use_cases"]
        except V1ValidationError as error:
            logger.error(log_prefix + f"error parsing file content, {error=}")
            return None

        return cases

    def save_license_use_cases(self, license_tag, license_use_cases):

        file_path = os.path.join(self.licenses_use_cases_dir, license_tag + ".yaml")
        write_yaml_file(file_path, license_use_cases)
        logger.info(f"file written: {file_path}")

    def read_licences_database(self):
        licenses_and_link = self.read_licenses_and_link()
        licenses_data = {}
        for item in licenses_and_link:
            tag = item["tag"]
            license_text = self.read_license_text(tag)
            license_str = "".join(license_text)
            if license_text is None:
                logger.error(f"error reading license text for {tag=}, skipping")
                continue
            license_use_cases = self.read_license_use_cases(tag)
            if (license_use_cases is None):
                logger.error(f"error reading license use cases for {tag=}, skipping")
                continue
            licenses_data[tag] = {
                "name": item["name"],
                "tag": tag,
                "text": license_str,
                "use_cases": license_use_cases,
            }
        return licenses_data
