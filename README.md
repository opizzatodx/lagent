# L'Agent

L'Agent is an open-source license assistant capable of providing verified answers to questions about the use of a license. 

This project was created for the [Generative AI Agents Developer Contest by NVIDIA and LangChain](https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/).

## Context
  
Understanding legal documents is not easy. They are written by experts, for experts, using expert vocabulary.
  
When used directly AI could not be trusted enough. For legal advices, there is no room for misinterpretations or hallucinations.

The project L’Agent  focuses on open-source licenses and provide verified answers to any question about them.  However, the concept and architecture can be applied to any contract or legal texts database.

## Solution

First l'Agent preprocesses licenses by generating all use cases related to each license, using a Large Language Model (LLM) with structured output.

These use cases can be verified and corrected by a legal expert.

L'Agent chat is a generative AI agent chatbot that uses only these certified use cases to respond to user requests.

L'Agent states its asumptions about which license and which use case is used to provide the answer, ensuring that the user is responsible for the shared understanding of the request, and the validity of the answer.

Finally the answer of L'Agent is cross-validated using [NEMO Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html) against the license name and the license use case, ensuring the trustfulness of the answer.

![Architecture of L'Agent.](./assets/images/lagent_architecture.png)

Chat workflow
  1. Initial question from the user.
  2. L'Agent interacts until the license is fully identified.
  3. Then L'Agent interacts until a use case of this license matches the user's use case.
  4. Then L'Agent provides the answer using this use case. The answer is cross-validated using [NEMO Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html).

L'Agent is preloaded with 32 open-source software licenses for demonstration purpose.

# Technologies used

* [NVIDIA NIM](https://build.nvidia.com/explore/discover) serves the LLM models used by L'Agent
  * LLM [mistral-large](https://build.nvidia.com/mistralai/mistral-large) for use case generation and chat engine
  * LLM [mixtral-8x22b-instruct](https://build.nvidia.com/mistralai/mixtral-8x22b-instruct) for cross-validation of L'Agent answer
* [NVIDIA NEMO Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html) for L'Agent output rail implementation
* [LangChain](https://www.langchain.com/langchain) as generative AI framework 
* [LangSmith](https://www.langchain.com/langsmith) for generative AI monitoring
* [Gradio](https://www.gradio.app/) as user interface 
        
# Design

## Use case generation

The use case generation is a LangChain chain with:
* a prompt with structured output build using [ChatPromptTemplate]() with license text plus [PydanticOutputParser]() instructions and output parsing
* a NVIDIA Chat LLM model using [```langchain-nvidia-ai-endpoints```](https://python.langchain.com/v0.2/docs/integrations/providers/nvidia/)

## Chat engine

The chat engine is a composition of 3 successive chats:

Each one is
* a [RunnableWithMessageHistory](https://python.langchain.com/v0.2/docs/how_to/message_history/) chain with
* a NVIDIA Chat LLM model using [```langchain-nvidia-ai-endpoints```](https://python.langchain.com/v0.2/docs/integrations/providers/nvidia/).

*LICENSE* chat
  * aims at identifying what is the exact license the user is talking about,
  * using a 3-shots CoT [ChatPromptTemplate]() with the list of license names as parameter.

*USE_CASE* chat
  * aims at identifying which use case from the use case database the user is talking about,
  * using a 2-shots CoT [ChatPromptTemplate]() with the list of the license use cases as parameter.
  * The LLM answer is prefixed with the name of the license that was identified by the *LICENSE* chat.

*RESPONSE* chat
  * aims at providing the final answer to the user,
  * using a 2-shots CoT [ChatPromptTemplate]() with the name of the license and the use case as parameters.
  * The LLM answer is prefixed with the name of the license and the use case that was identified in by the *LICENSE* and *USE_CASE* chats.
  * The LLM model is decorated with [NEMO Guardrails RunnableRails](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/langchain/runnable-rails.md),
      * configured with an [output rail](https://docs.nvidia.com/nemo/guardrails/getting_started/5_output_rails/README.html) with the license name and use case as [NEMO Guardrails prompt variables](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/advanced/prompt-customization.md#prompt-variables)
      * prompted to verify the consistency of the LLM output with the license name and use case.


# Install

* Define environment variables inside a .env file

```sh
# Environment variables for LANG SMITH
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=

# Environment variables for NVIDIA NIM API
NVIDIA_API_KEY=
```

* Setup a virtual env

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements
```

* Launch the Gradio app
  * '-s' option allows to share the local gradio app with others

```sh
python app.py -h
```

* Use the command line interface (CLI) to: 
  * list available licenses
  * generating use cases for a license
  * running the L'Agent chat in command line

```sh
python cli.py -h
```
```
Command line interface for L'Agent.

positional arguments:
  {license,use_case,chat}
    license             License management
    use_case            Use case management
    chat                Chat with L'Agent.
```


# Usage

## Chat app

Launch the Chat app
```sh
python app.py -s
```

The Chat app has two tabs:
* The chat interface: You can chat about your use of a specific license.
  * For example, you can ask: 'Can I use GPL for personal use?'
  * You must clear the chat history ("clear" button) to start a new conversation and reset the context from the previous one.

![Chat tab of L'Agent.](./assets/images/lagent_chat.png)

* The license database: Displays all the licenses supported by the chat.
  * Select a license in the table to see its content and the list of use cases.

![Licenses tab of L'Agent.](./assets/images/lagent_licenses.png)

## CLI for preprocessing 

* Run the use case generator on a specific license

```sh
python cli.py use-case generate -t cecill-2.1
```

* Licenses are listed in the file 

```sh
./data/licenses_and_link.yaml
```

Note: This file is generated by OpenAI ChatGPT 4o with the following prompt:

```text
List all open source licenses, for each provide a link to the license text and a unique tag, output format must be yaml
```

* You can add a new license by modifying this file
* License contents are in the directory:

```sh
./data/licenses_text
```

* You can add the content of a new license in this directory, file name format is ```[license_tag].txt ```

* You can run the use case generator on all licenses:

```sh
python cli.py use-case generate -a
```

* You can list all license tags:

```sh
python cli.py license list
```

## Acknowledgments

Many thanks to [NVIDIA](https://www.nvidia.com/fr-fr/) and [LangChain](https://www.langchain.com/) for this opportunity and the support given during the contest.

Special thanks to my company [Darwin-X](https://www.darwin-x.com/) for allowing me to invest time in this project.

## Contributing

PRs accepted.

## License

MIT © Olivier Pizzato