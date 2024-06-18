import yaml
import logging
import os

logger = logging.getLogger(__name__)


def find_a_line_starting_with(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def write_yaml_file(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def read_yaml_file(file_path, skip_root_tag=None):
    if (not os.path.exists(file_path)):
        logger.error(f"file {file_path} does not exist.")
        return None
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    if skip_root_tag is not None:
        data = data.get(skip_root_tag, None)
    return data


def check_data_format_as_list_of_dicts_with_keys(data, keys, log_prefix=""):
    if not isinstance(data, list):
        return False, log_prefix + "data must be a list of dictionaries."
    for item in data:
        if not isinstance(item, dict):
            return False, log_prefix + f"each item of the list must be a dictionary, found: {item}."
        for key in keys:
            if key not in item:
                return False, log_prefix + f"each item of the list must have keys: {keys}, found: {item}."
    return True, None


def load_text_file_lines(file_path):
    if (not os.path.exists(file_path)):
        logger.error(f"file {file_path} does not exist.")
        return None

    with open(file_path, "r", encoding='utf-8') as file:
        return file.readlines()
