from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, PromptTemplate, \
    HumanMessagePromptTemplate
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


def asking_and_getting_answers(vectorstore, question: str, chat_history):
    # llm
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, temperature=0.5)

    # prompt
    prompt = build_chat_prompt_template()

    # chain
    chain = create_stuff_documents_chain(llm, prompt)

    # retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # retriever prompt
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    # retrieval chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    resp = retrieval_chain.invoke({"input": question, "chat_history": chat_history, 'context': retriever})
    _input = resp['input']
    output = resp['answer']
    chat_history.append(HumanMessage(content=_input))
    chat_history.append(AIMessage(content=output))

    return output


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
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
    )
