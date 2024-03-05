from operator import itemgetter

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_vectorstore(index_name, encoding):
    return PineconeVS.from_existing_index(index_name, encoding)


def get_retriever(vs):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0.2,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def get_chat_prompt() -> ChatPromptTemplate:
    system_template = (
        """
        Você é um assistente de negócios. Você está auxiliando um gerente de contas de um banco. 
        Considere o histórico de mensagens a seguir, crie uma consulta de busca para encontrar informações sobre transações. 
        Só utilize os documentos fornecidos como fonte de resposta se a pergunta for sobre transações bancárias.
        Se não souber uma resposta, apenas responda "Não tenho informação suficiente para responder sobre isso". 
        Responda, por padrão, de maneira simples e direta.
        """
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history'),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
    )


def get_stuff_docs_prompt() -> ChatPromptTemplate:
    system_template = (
        """
        Os documentos relacionados são referentes a transações bancárias dos clientes da carteira de um gerente de contas. 
        É possível encontrar informações sobre transações avulsas e agrupadas. Aqui um exemplo de transação avulsa: 
        
        < transação avulsa > 
        O cliente Carlos Eduardo Rodrigues(CPF: 80763159212), efetuou um transação de R$ 618.81 no dia 2 do mês de 
        março do ano de 2023(02 / 03 / 2023) com cartão de débito BLACK para o estabelecimento "Azevedo S/A" 
        < / transação avulsa > 
        
        As agrupadas contém informação por dia, por mês e por ano. Aqui um exemplo de informação sobre transações agrupadas: 
        
        < informação sobre transações agrupadas >
        Sumário mensal de transações do cliente Carlos Eduardo Rodrigues(CPF: 80763159212) com cartão de débito para o 
        mês de março do ano de 2023(03 / 2023): - Contagem de transações: 25, (4 com cartão BLACK, 4 com cartão GOLD, 4 
        com cartão PLATINUM, 5 com cartão STANDARD, 8 com cartão INTERNACIONAL); - Valor total: R$ 62126.99; - Valor médio: R$ 2485.08; 
        - Valor da maior transação: R$ 4809.23; - Valor da menor transação: R$ 47.65. 
        < / informação sobre transações agrupadas >
        
        Utilize o contexto fornecido para responder perguntas ligadas a transações bancárias. 
        
        < contexto > 
        {context} 
        < / contexto >  
        
        Questão: {input}
        """
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history'),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
    )


def get_buffer_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.load_memory_variables({})
    return memory


def build_rag_chain():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    retriever = get_retriever(vs)
    llm = get_llm()

    # cria chain que utiliza histórico de conversa pra retornar documentos
    retriever_chain = create_history_aware_retriever(llm=llm,
                                                     retriever=retriever,
                                                     prompt=get_chat_prompt())

    # cria chain de automação que vai organizar uma lista de documentos
    document_chain = create_stuff_documents_chain(llm=llm,
                                                  prompt=get_stuff_docs_prompt())

    # cria chain que combina os dois anteriores
    conversation_retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

    memory = get_buffer_memory()

    pass_through = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter('chat_history')
    )

    chain = pass_through | conversation_retriever_chain
    return conversation_retriever_chain, memory, chain


def ask_rag_chain(question):
    retriever, memory, rag_chain = build_rag_chain()
    response = rag_chain.invoke({"input": question, "context": retriever})
    memory.save_context({"input": question}, {"output": response["answer"]})
    memory.load_memory_variables({})
    return response["answer"]
