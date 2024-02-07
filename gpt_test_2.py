import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 1: Carrega o arquivo
file = client.files.create(
  file=open("texto.txt", "rb"),
  purpose='assistants'
)

# 2: Cria o assistente
chat_assistant = client.beta.assistants.create(
    model="gpt-4-turbo-preview",
    name="Comercial Manager Support Chatbot",
    instructions="Você é um assistente, operando no dia 2024-02-07 (formato yyyy-MM-dd), que responde em portugues. "
                 "O usuário enviará um texto que representa algumas transações financeiras e fará algumas perguntas sobre elas. "
                 "Utilize respostas curtas. "
                 "O arquivo contém valores separado por pipe referentes ao seguinte cabeçalho: Data da transação, Descrição, Tipo do cartão, Valor, Nome do cliente, Número do documento do cliente(CPF). "
                 "Todas as datas utilizadas estão no formato yyyy-MM-dd.",
    tools=[{"type": "retrieval"}]
)

# 3: Cria a thread
thread = client.beta.threads.create()

while True:
    user_input = input("\nDigite sua pergunta (ou 'sair' para terminar): ")
    if user_input.lower() == 'sair':
        break

    # 4: Adiciona a mensagem na thread
    thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
        file_ids=[file.id]
    )

    # 5: roda o assistente
    my_run = client.beta.threads.runs.create(
      thread_id=thread.id,
      assistant_id=chat_assistant.id
    )

    # 6: Fica escutando o status para gatilho do comportamento
    while my_run.status in ["queued", "in_progress"]:
        keep_retrieving_run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=my_run.id
        )

        if keep_retrieving_run.status == "completed":
            # 7: pega as mensagens que foram adicionadas a thread
            all_messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            print("------------------------------------------------------------ \n")
            print(f"User: {thread_message.content[0].text.value}")
            print(f"Assistant: {all_messages.data[0].content[0].text.value}")

            break