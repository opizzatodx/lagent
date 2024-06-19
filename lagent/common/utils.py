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


def load_text_file_lines(file_path):
    if (not os.path.exists(file_path)):
        logger.error(f"file {file_path} does not exist.")
        return None

    with open(file_path, "r", encoding='utf-8') as file:
        return file.readlines()
