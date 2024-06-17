import pandas as pd
import gradio as gr
import logging
import argparse
from dotenv import load_dotenv
from lagent.chat.agent import LicenseAgent
from lagent.licenses.licenses import LicensesProcessor

# https://github.com/langchain-ai/langchain/discussions/21596
logging.getLogger("langchain_core.tracers.core").setLevel(logging.ERROR)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Launch the Gradio app chatbot")
    parser.add_argument("-s", "--share", action="store_true", help="to share the app with a public link")
    return parser.parse_args()


def create_licenses_dataframe(licenses_data):
    data = []
    for tag, item in licenses_data.items():
        data.append([tag, item["name"], len(item["text"]), len(item["usage_cases"])])
    df = pd.DataFrame(data, columns=["tag", "name", "content length", "number of usage cases"])
    df = df.sort_values(by="tag", ascending=True)
    return df

# license data
licenses_processor = LicensesProcessor(
    licenses_and_link_file_path="./data/licenses_and_link.yaml", 
    licenses_text_dir="./data/licenses_text", 
    licenses_usage_cases_dir="./data/usage_cases"
)
licenses_data = licenses_processor.read_licences_database()
licenses_df = create_licenses_dataframe(licenses_data)
license_list = licenses_df["name"].values.tolist()
usage_cases_table_columns = ["allowed", "title", "description", "conditions"]

# agent instance
agent = LicenseAgent(licenses_data)


# chatbot callback function
def chat_response(message, history):
    logger.info(f"{len(history)=}, processing message: {message=}")
    return agent.chat(message, history)


# license selection callback function
def on_select_a_license(selected_index: gr.SelectData):

    # get the selected license tag
    index = selected_index.index[0]
    tag = licenses_df.iloc[index]["tag"]

    # get the license content
    license_content = licenses_data[tag]["text"]

    # get the usage cases as dataframe
    license_usage_cases = licenses_data[tag]["usage_cases"]
    df = pd.DataFrame(license_usage_cases)
    df = df[usage_cases_table_columns]

    return [license_content, df]


def main(args):

    chatbot = gr.Chatbot(elem_id="chatbot", height=600)
    with gr.Blocks() as chat_blocks:
        notice = gr.Markdown("""
            Chat about your usage of a __specific license__. For example, you can ask: 'Can I use GPL for personal use?'\n
            __Clear the chat history__ to start a new conversation. 
        """)
        chat = gr.ChatInterface(
            fn=chat_response,
            chatbot=chatbot,
            undo_btn=None,
            retry_btn=None,
        )

    with gr.Blocks() as license_blocks:
        notice = gr.Markdown("""
            Licenses supported by the chat. __Select a license__ in the table below to see its content and the list of usage cases.\n
        """)
        with gr.Row():
            licenses_table = gr.Dataframe(value=licenses_df)
        with gr.Row():
            license_content = gr.Textbox(label="License content")
            usage_cases_table = gr.Dataframe(label="Usage cases", value=pd.DataFrame(columns=usage_cases_table_columns))

        licenses_table.select(on_select_a_license, None, [license_content, usage_cases_table])

    demo = gr.TabbedInterface([chat_blocks, license_blocks], ["Chat", "Licenses"])
    demo.launch(share=args.share)


if __name__ == "__main__":
    args = parse_args()
    main(args)

