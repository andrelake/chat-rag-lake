from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        chat_history = []

    # retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # system prompt
    contextualize_q_prompt = system_prompt_handler()

    # llm
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, temperature=0.5)

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    rag_chain = (
            RunnablePassthrough.assign(context=contextualize_q_chain | retriever)
            | contextualize_q_prompt
            | llm
    )

    resp = rag_chain.invoke({"question": question, "chat_history": chat_history, "context": retriever})
    chat_history.extend([HumanMessage(content=question), AIMessage(content=resp.content)])

    return resp.content, chat_history


def system_prompt_handler():
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    # prompt
    contextualize_q_system_prompt = (
        "Você deve entender e dar todas as respostas em português, do Brasil. "
        "Você é um assistente que responderá perguntas com base no contexto abaixo. "
        "O contexto é um fórum de bate-papo que tem a coordenação geral do Ministro Chefe da Casa Civil, "
        "com a coordenação executiva do Ministro de Estado da Educação e a participação efetiva e estratégica "
        "das Empresas Estatais brasileiras. "
        "Combine o histórico de bate-papo e a última pergunta do usuário em "
        "uma pergunta independente que pode ser entendida sem o histórico de bate-papo. "
        "Contexto: {context} "
        "Histórico de bate-papo: {chat_history} "
        "Pergunta de acompanhamento: {question}"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    return contextualize_q_prompt
