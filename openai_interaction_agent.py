from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

from env import OPENAI_API_KEY, OPENAI_MODEL_NAME


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
        chat_history = ChatMessageHistory()

    # retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # history aware retriever


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
    # memory = VectorStoreRetrieverMemory(
    #     memory_key="chat_history",
    #     return_docs=True,
    #     retriever=retriever,
    #     return_messages=True,
    #     )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        input_key='input',
        return_messages=True)

    # llm
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, temperature=0.5)

    # agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=chat_prompt_template,
    )

    # agent_executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

    resp = agent_executor.invoke({"input": question, "chat_history": chat_history})
    _input = resp.get('input')
    output = resp.get('output')
    chat_history.add_user_message(_input)
    chat_history.add_ai_message(output)

    return output, chat_history


def build_chat_prompt_template() -> ChatPromptTemplate:
    system_template = (
        "You are a helpful assistant. "
        "You can only use Portuguese, from Brazil, to read and answer all the questions. "
        "You can use the following pieces of context to answer the question. {context} "
        # "Combine chat history and the last user's question to create a new independent question. "
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'chat_history', 'question'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history', optional=True),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], template='{question}')),
         MessagesPlaceholder(variable_name='agent_scratchpad')]
    )
    # return ChatPromptTemplate.from_messages(
    #             [
    #                 ("system", system_template),
    #                 MessagesPlaceholder(variable_name="chat_history"),
    #                 ("human", "{input}"),
    #                 MessagesPlaceholder("agent_scratchpad"),
    #             ]
    #         )
