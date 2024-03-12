from operator import itemgetter

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key,
                            model="text-embedding-3-small")


def get_vectorstore(index_name, encoding):
    return PineconeVS.from_existing_index(index_name, encoding)


def get_retriever(vs):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def get_chat_prompt() -> ChatPromptTemplate:
    system_template = (
        """
        Você é um assistente de negócios. Você está auxiliando um gerente de contas de um banco. 
        Considere o histórico de mensagens a seguir, crie uma consulta de busca para encontrar informações sobre transações de cartão de crédito. 
        Só utilize os documentos fornecidos como fonte de resposta se a pergunta for sobre transações bancárias.
        Considere utilizar o histórico de mensagens para criar uma consulta de busca ou responder perguntas.  
        Se não souber uma resposta, apenas responda "Não tenho informação suficiente para responder sobre isso". 
        Responda, por padrão, de maneira simples e direta.
        
        Os documentos relacionados são referentes a transações bancárias dos clientes da carteira de um gerente de contas. 
        É possível encontrar sumários diários, mensais e anuais para cada cliente, existem também sumários diários, mensais e anuais de toda a carteira de clientes.
        Desde que responda corretamente a pergunta, dê preferência a extrair informações dos sumários mais assertivos para a pergunta, por exemplo, 
        se a pergunta for sobre um determinado mês busque informações no sumário mensal ao invés de somar todos os sumários diários daquele mês, 
        ou se a pergunta for sobre a carteira (todos os clientes) busque informações nos sumários da carteira ao invés de somar os sumários de cada cliente.
        
        Exemplo de sumário diário de um cliente:
        <doc> 
        [Quanto o cliente Ana Lívia da Cruz (CPF: 49738156084) transacionou com cartão de crédito no dia 2 do mês de janeiro do ano de 2023 (02/01/2023)?]	Resumo diário das transações do cliente no dia: 	R$ 1095.13 no estabelecimento "Moreira";	R$ 2017.50 no estabelecimento "da Paz S/A";
        </doc> 
        
        Exemplo de sumário mensal da carteira:
        <doc> 
        [Qual o total transacionado com cartão de crédito por todos os clientes no mês de julho do ano de 2023 (07/2023)?]	Resumo mensal de todas as transaçoes no mês, com uma contagem total de 279 transações e valor total de R$ 706888.44, o valor médio é de R$ 2533.65, o valor da maior transação é de R$ 4999.94 e o valor da menor transação é de R$ 26.86. Dentre elas foram realizadas 55 com cartão BLACK, 62 com cartão GOLD, 59 com cartão PLATINUM, 51 com cartão STANDARD, 52 com cartão INTERNACIONAL.
        </doc>
        
        Questão: {input}
        """
    ).replace(' ' * 4, '')

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history'),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
    )


def get_stuff_docs_prompt() -> ChatPromptTemplate:
    system_template = (
        """
        Utilize o contexto fornecido e o histórico de mensagens para responder perguntas ligadas a transações bancárias. 
        
        <contexto> 
        {context} 
        </contexto>  
        
        Questão: {input}
        """
    ).replace(' ' * 4, '')

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

    document_prompt = PromptTemplate(input_variables=["page_content", "consumer_name"],
                                     template="{page_content}, consumer_name: {consumer_name}")

    # cria chain de automação que vai organizar uma lista de documentos
    document_chain = create_stuff_documents_chain(llm=llm,
                                                  prompt=get_stuff_docs_prompt(),
                                                  document_prompt=document_prompt)

    # cria chain que combina os dois anteriores
    conversation_retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

    memory = get_buffer_memory()

    pass_through = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter('chat_history')
    )

    chain = pass_through | conversation_retriever_chain
    return retriever, conversation_retriever_chain, memory, chain


def ask_rag_chain(question, conversation_retriever_chain, memory, rag_chain):
    # return retriever.invoke(question)
    response = rag_chain.invoke({"input": question, "context": conversation_retriever_chain})
    memory.save_context({"input": question}, {"output": response["answer"]})
    memory.load_memory_variables({})
    return response["answer"]
