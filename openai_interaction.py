from langchain.agents import AgentExecutor

from env import OPENAI_MODEL_NAME

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool


def get_interaction(key: str) -> None:
    # llm
    llm = ChatOpenAI(openai_api_key=key, model_name=OPENAI_MODEL_NAME, temperature=0)

    # chat template
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessagePromptTemplate.from_template("{instruction}"),
    ])

    # caching
    set_llm_cache(InMemoryCache())

    # chain
    chain = LLMChain(llm=llm, prompt=chat_template, verbose=True)

    # streaming
    # for chunk in llm.stream(messages):
    #     print(chunk.content, end='', flush=True)

    # agent
    agent_executor: AgentExecutor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True
    )

    instruction = input("Enter instruction: ")
    resp = agent_executor.invoke({"input": instruction})
    # resp = chain.invoke({"instruction": instruction})
    print(agent_executor.tools[0].name, resp.get('output'))
