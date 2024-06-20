import logging
from dotenv import load_dotenv
from lagent.chat.agent import Agent
from lagent.licenses.licenses import LicenseProcessor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Command line interface for L\'Agent.')
    subparsers = parser.add_subparsers(dest="action")

    # license management
    license_parser = subparsers.add_parser('license', help='License management')

    license_subparsers = license_parser.add_subparsers(dest="sub_action")
    license_subparsers.add_parser('list', help='list licenses')

    use_case_parser = subparsers.add_parser('use_case', help='Use case management')
    use_case_subparsers = use_case_parser.add_subparsers(dest="sub_action")

    # generate use cases
    generate_uc_parser = use_case_subparsers.add_parser('generate', help='generate use cases for licenses')
    generate_uc_parser.add_argument('-a', '--all', action='store_true', help='generate use cases for all licenses')
    generate_uc_parser.add_argument('-t', '--license_tag', type=str, help='generate use cases for the [license_tag] license')

    # chat command line
    chat_parser = subparsers.add_parser('chat', help='Chat with L\'Agent. ')
    chat_parser.add_argument('-q', '--query', required=True, help='chat user query. "quit" to stop the chat loop. "clear" to clear the chat history.')

    args = parser.parse_args()
    return args


def main(args):

    logger.info(f"{args=}")

    license_processor = LicenseProcessor(
        licenses_and_link_file_path="./data/licenses_and_link.yaml", 
        licenses_text_dir="./data/licenses_text", 
        licenses_use_cases_dir="./data/use_cases"
    )
    licenses = license_processor.read_licenses_and_link()
    logger.info(f"{len(licenses)=}")

    if args.action == "license":
        if args.sub_action == "list":
            for license in licenses:
                print(license["tag"])
            return

    if args.action == "use_case":
        if args.sub_action == "generate":

            # possibly restrict to one license
            if args.license_tag:
                licenses = [d for d in licenses if d["tag"] == args.license_tag]
                if len(licenses) == 0:
                    logger.error(f"license tag not found: {args.license_tag}")
                    return

            for license in licenses:
                tag = license["tag"]
                use_cases = license_processor.generate_license_use_cases(tag)
                license_processor.save_license_use_cases(tag, use_cases)

    if args.action == "chat":

        licenses_data = license_processor.read_licences_database()
        if licenses_data is None:
            logger.error("error reading licenses database")
            return False

        agent = Agent(licenses_data)
        while args.query != "quit":
            res = agent.chat_for_matching_the_license(args.query)
            print(res)
            args.query = input("Your query: ")



if __name__ == "__main__":
    args = parse_args()
    main(args)
