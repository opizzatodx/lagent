import pandas as pd
import gradio as gr
import logging
import argparse
from dotenv import load_dotenv
from lagent.chat.agent import Agent
from lagent.licenses.licenses import LicenseProcessor

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
        data.append([tag, item["name"], len(item["text"]), len(item["use_cases"])])
    df = pd.DataFrame(data, columns=["tag", "name", "content length", "number of use cases"])
    df = df.sort_values(by="tag", ascending=True)
    return df


class App:
    def __init__(self, license_processor, licenses_data, license_agent):
        self.processor = license_processor
        self.licenses_data = licenses_data
        self.license_agent = license_agent

        self.licenses_df = create_licenses_dataframe(self.licenses_data)
        self.use_cases_table_columns = ["allowed", "title", "description", "conditions"]

    # chat callback function
    def chat(self, message, history):    

        logger.info(f"{len(history)=}, processing message: {message=}")
        return self.license_agent.chat(message, history)

    # license selection callback function
    def on_select_a_license(self, selected_index: gr.SelectData):

        # get the selected license tag
        index = selected_index.index[0]
        tag = self.licenses_df.iloc[index]["tag"]

        # get the license content
        license_content = self.licenses_data[tag]["text"]

        # get the use cases as dataframe
        license_use_cases = self.licenses_data[tag]["use_cases"]
        df = pd.DataFrame(license_use_cases)
        df = df[self.use_cases_table_columns]

        return [license_content, df]

    def launch(self, share=False):

        chatbot = gr.Chatbot(elem_id="chatbot", height=600)
        with gr.Blocks() as chat_blocks:
            gr.Markdown("""
                Chat about your use of a __specific license__. For example, you can ask: 'Can I use GPL for personal use?'\n
                __Clear the chat history__ to start a new conversation. 
            """)
            gr.ChatInterface(
                fn=self.chat,
                chatbot=chatbot,
                undo_btn=None,
                retry_btn=None,
            )

        with gr.Blocks() as license_blocks:
            gr.Markdown("""
                Licenses supported by the chat. __Select a license__ in the table below to see its content and the list of use cases.\n
            """)
            with gr.Row():
                licenses_table = gr.Dataframe(value=self.licenses_df)
            with gr.Row():
                license_content = gr.Textbox(label="License content")
                use_cases_table = gr.Dataframe(label="Use cases", value=pd.DataFrame(columns=self.use_cases_table_columns))

            licenses_table.select(self.on_select_a_license, None, [license_content, use_cases_table])

        app = gr.TabbedInterface([chat_blocks, license_blocks], ["Chat", "Licenses"])
        app.launch(share=share)


def main(args):

    license_processor = LicenseProcessor(
        licenses_and_link_file_path="./data/licenses_and_link.yaml", 
        licenses_text_dir="./data/licenses_text", 
        licenses_use_cases_dir="./data/use_cases"
    )
    licenses_data = license_processor.read_licences_database()
    if licenses_data is None:
        logger.error("error reading licenses database")
        return False

    agent = Agent(licenses_data)

    app = App(license_processor, licenses_data, agent)
    app.launch(share=args.share)


if __name__ == "__main__":
    args = parse_args()
    main(args)

