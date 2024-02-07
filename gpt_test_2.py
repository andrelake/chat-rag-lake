import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import date
from time import sleep
from env import OPENAI_API_KEY


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    END = '\033[0m'


client = OpenAI(api_key=OPENAI_API_KEY)

# 1: Carrega o arquivo
file = client.files.create(
    file=open("texto.txt", "rb"),
    purpose='assistants'
)

manager_name = 'Fernando Dias'
current_date = date.today().strftime('%Y-%m-%d')
date_format = 'yyyy-MM-dd'
headers = ['Data', 'Descrição', 'Tipo', 'Valor', 'Nome do cliente', 'CPF do cliente']
instructions = f'''
    Você é o assistente do gerente de contas {manager_name} da PicPay que opera em uma determinada carteira de clientes.
    Você receberá um arquivo PSV (separado por pipe) contendo todas as transações das faturas de cartões (crédito e débito) dos clientes dessa carteira, essas são as colunas do cabeçalho: {', '.join(headers)}.
    Utilize respostas curtas e cálcule com 100% de precisão.
    Hoje é dia {current_date}.
    Todas as datas no arquivo estão no formato {date_format}.
'''.replace('\n' + ' '*4, '\n').strip()

# 2: Cria o assistente
chat_assistant = client.beta.assistants.create(
    model="gpt-4-turbo-preview",
    name="Comercial Manager Support Chatbot",
    instructions=instructions,
    tools=[{"type": "retrieval"}]
)

# 3: Cria a thread
thread = client.beta.threads.create()

block_separator = Color.CYAN + '-' * 80 + Color.END
print(block_separator, end='\n\n')
print(f'{Color.CYAN}>>> System:{Color.END} {instructions}')
print(f'{Color.CYAN}>>> System: `texto.txt`{Color.END}', end='\n\n')


while True:
    print(block_separator, end='\n\n')
    user_input = input(f'{Color.CYAN}>>> Digite sua pergunta (ou "sair"):{Color.END} ')
    if user_input.lower() == 'sair':
        break

    print(f'\033[1A{Color.CYAN}>>> User:{Color.END} {user_input}\033[K') # thread_message.content[0].text.value
    
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
            all_messages = client.beta.threads.messages.list(thread_id=thread.id)

            print(f'{Color.CYAN}>>> Assistant:{Color.END} {all_messages.data[0].content[0].text.value}', end='\n\n')
            break

        sleep(1)
