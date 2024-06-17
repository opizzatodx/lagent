import logging
from dotenv import load_dotenv
from lagent.licenses.licenses import LicensesProcessor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Rewrite one or each license into an complete list of usage cases.')
    parser.add_argument('-a', '--all', action='store_true', help='process all licenses')
    parser.add_argument('-t', '--license_tag', type=str, help='license tag to process, process one license')
    parser.add_argument('-l', '--list', action='store_true', help='list all license tags')
    args = parser.parse_args()
    return args


def main(args):

    logger.info(f"{args=}")

    licenses_processor = LicensesProcessor(
        licenses_and_link_file_path="./data/licenses_and_link.yaml", 
        licenses_text_dir="./data/licenses_text", 
        licenses_usage_cases_dir="./data/usage_cases"
    )
    licenses = licenses_processor.read_licenses_and_link()
    logger.info(f"{len(licenses)=}")

    if args.list:
        for license in licenses:
            print(license["tag"])
        return

    if (args.all and args.license_tag) or (not args.all and not args.license_tag):
        logger.error("either --all or --license_tag must be specified, but not both")
        return

    # possibly restrict to one license
    if args.license_tag:
        licenses = [d for d in licenses if d["tag"] == args.license_tag]
        if len(licenses) == 0:
            logger.error(f"license tag not found: {args.license_tag}")
            return

    for license in licenses:
        tag = license["tag"]
        usage_cases = licenses_processor.generate_license_usage_cases(tag)
        licenses_processor.save_license_usage_cases(tag, usage_cases)

if __name__ == "__main__":
    args = parse_args()
    main(args)
