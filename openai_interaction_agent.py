from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

from env import OPENAI_API_KEY, OPENAI_MODEL_NAME, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN


# def get_interaction(key: str) -> None:
#     # llm
#     llm = ChatOpenAI(openai_api_key=key, model_name=OPENAI_MODEL_NAME, temperature=0)
#
#     # chat template
#     chat_template = ChatPromptTemplate.from_messages([
#         SystemMessage(content="You are a helpful assistant."),
#         HumanMessagePromptTemplate.from_template("{instruction}"),
#     ])
#
#     # caching
#     set_llm_cache(InMemoryCache())
#
#     # chain
#     chain = LLMChain(llm=llm, prompt=chat_template, verbose=True)
#
#     # streaming
#     # for chunk in llm.stream(messages):
#     #     print(chunk.content, end='', flush=True)
#
#     # agent
#     agent_executor: AgentExecutor = create_python_agent(
#         llm=llm,
#         tool=PythonREPLTool(),
#         verbose=True
#     )
#
#     instruction = input("Enter instruction: ")
#     resp = agent_executor.invoke({"input": instruction})
#     # resp = chain.invoke({"instruction": instruction})
#     print(agent_executor.tools[0].name, resp.get('output'))


def asking_and_getting_answers(vectorstore, question: str, chat_history=None):
    if chat_history is None:
        chat_history = UpstashRedisChatMessageHistory(
            url=UPSTASH_REDIS_REST_URL,
            token=UPSTASH_REDIS_REST_TOKEN,
            session_id='session_id'
        )

    # retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # tool
    tool = create_retriever_tool(
        retriever=retriever,
        name="search_state_forum_of_education",
        description="Searches and returns content about the State Forum of Education.",
    )
    tools = [tool]

    # chat_prompt_template
    chat_prompt_template = build_chat_prompt_template()

    # memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        input_key='input',
        chat_memory=chat_history,
        return_messages=True)

    # llm
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name=OPENAI_MODEL_NAME,
                     temperature=0.5,
                     streaming=True)

    # agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=chat_prompt_template,
    )

    # agent_executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

    resp = agent_executor.invoke({"input": question, "chat_history": chat_history, 'context': retriever})

    return resp.get('output')


def build_chat_prompt_template() -> ChatPromptTemplate:
    system_template = (
        "You are a helpful assistant. "
        "You can only use Portuguese, from Brazil, to read and answer all the questions. "
        "You can use the following pieces of context to answer the question. {context} "
        "Combine chat history and the last user's question to create a new independent question. "
        "Histórico de bate-papo: {chat_history} "
        "Pergunta de acompanhamento: {input}"
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'chat_history', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history', optional=True),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
         MessagesPlaceholder(variable_name='agent_scratchpad')]
    )
