import logging
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from lagent.common.utils import check_data_format_as_list_of_dicts_with_keys, load_text_file_lines, read_yaml_file, write_yaml_file
from lagent.licenses.usage import prompt, parser

logger = logging.getLogger(__name__)


class LicensesProcessor():

    def __init__(self, licenses_and_link_file_path, licenses_text_dir, licenses_usage_cases_dir):

        self.licenses_and_link_file_path = licenses_and_link_file_path
        self.licenses_text_dir = licenses_text_dir
        self.licenses_usage_cases_dir = licenses_usage_cases_dir

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
        keys = ["tag", "name"]
        res, error = check_data_format_as_list_of_dicts_with_keys(data, keys, log_prefix=f"read ${self.licenses_and_link_file_path}: ")
        if not res:
            logger.error(f"error reading licenses file, {error=}")
            return None
        
        return data
    
    def read_license_text(self, license_tag):
        file_path = os.path.join(self.licenses_text_dir, license_tag + ".txt")
        data = load_text_file_lines(file_path)
        if (data is None):
            logger.error(f"error reading license text for {license_tag=}")
            return None
        data_str = "".join(data)
        logger.info(f"read license file {file_path=}, {len(data_str)=}")    
        return data

    def generate_license_usage_cases(self, license_tag):

        license_content = self.read_license_text(license_tag)
        logger.info(f"processing license: {license_tag}, {len(license_content)=}")
        license_str = "".join(license_content)

        try:
            result = self.chain.invoke({"licence_text": license_str})
        except Exception as e:
            logger.error(f"error processing license: {license_tag}, {e=}")
            return None

        logger.info(f"{len(result.usage_cases)=}")
        for uc in result.usage_cases:
            logger.info(f"{uc.title=}")

        res = result.dict()
        return res

    def read_license_usage_cases(self, license_tag):

        file_path = os.path.join(self.licenses_usage_cases_dir, license_tag + ".yaml")
        data = read_yaml_file(file_path=file_path, skip_root_tag="usage_cases")
        if (data is None):
            logger.error(f"error reading license usage cases file {file_path=}")
            return None
        logger.info(f"read license usage cases file {file_path=}, {len(data)=}")
        keys = ["title", "description", "allowed", "conditions"]
        res, error = check_data_format_as_list_of_dicts_with_keys(data, keys, log_prefix=f"read ${file_path}: ")
        if not res:
            logger.error(f"error reading license usage cases file, {error=}")
            return None
        return data

    def save_license_usage_cases(self, license_tag, license_usage_cases):

        file_path = os.path.join(self.licenses_usage_cases_dir, license_tag + ".yaml")
        write_yaml_file(file_path, license_usage_cases)
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
            license_usage_cases = self.read_license_usage_cases(tag)
            if (license_usage_cases is None):
                logger.error(f"error reading license usage cases for {tag=}, skipping")
                continue
            licenses_data[tag] = {
                "name": item["name"],
                "tag": tag,
                "text": license_str,
                "usage_cases": license_usage_cases,
            }
        return licenses_data
