from datetime import date
from env import OPENAI_API_KEY
from re import sub

from old.connections import SQLite

from openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    END = '\033[0m'

db_langchain = SQLDatabase.from_uri('sqlite:///database.db')
db = SQLite()
db.terraform_db()

class Session:
    divider_str = Color.CYAN + '-' * 80 + Color.END
    date_format = 'yyyy-MM-dd'
    headers = ('ID da transação', 'Data', 'Descrição', 'Tipo', 'Valor', 'Nome do cliente', 'CPF do cliente')

    def __init__(self, openai_api_key: str, manager_name: str, current_date: date):
        self.llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo', temperature=0.2, max_tokens=300, top_p=0.1)
        self.sql_agent = create_sql_agent(self.llm, db=db, agent_type="openai-tools", verbose=True)
        self.client = OpenAI(api_key=openai_api_key)
        self.thread = self.client.beta.threads.create()
        self.assistant = None
        self.instructions = None
        self.current_date = current_date
        self.manager_name = manager_name
        self.instructions = sub(
            ' +', ' ',
            f'''
                Você é um assistente impecável de {self.manager_name}, um gerente de contas que opera em uma determinada carteira de clientes da PicPay.
                Você receberá um arquivo PSV (separado por pipe) contendo todas as transações das faturas de cartões (crédito e débito) dos clientes dessa carteira, essas são as colunas do cabeçalho: {', '.join(self.headers)}.
                Utilize respostas curtas e cálcule com 100% de precisão.
                Hoje é dia {self.current_date.strftime('%Y-%m-%d')}.
                Todas as datas no arquivo estão no formato {self.date_format}.
            '''.strip()
        )
    
    # def load_file(self):
    #     file = self.client.files.create(
    #         file=open(self.file_path, "rb"),
    #         purpose='assistants'
    #     )
    #     return file
    
    # def create_assistant(self) -> None:
    #     assistant = self.client.beta.assistants.create(
    #         model="gpt-4-turbo-preview",
    #         name="Comercial Manager Support Chatbot",
    #         instructions=self.instructions,
    #         tools=[{"type": "retrieval"}]
    #     )
    #     self.assistant = assistant
    
    # def add_message(self, message: str) -> None:
    #     thread_message = self.client.beta.threads.messages.create(
    #         thread_id=self.thread.id,
    #         role="user",
    #         content=message,
    #         file_ids=[self.load_file().id]
    #     )
    #     self.thread_message = thread_message
    
    # def run_assistant(self):
    #     my_run = self.client.beta.threads.runs.create(
    #         thread_id=self.thread.id,
    #         assistant_id=self.assistant.id
    #     )
    #     return my_run
    
    def get_messages(self) -> None:
        all_messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        self.all_messages = all_messages
    
    def ask(self, question: str) -> None:
        self.sql_agent.invoke(input=question)

    @staticmethod
    def print_divider() -> None:
        print(Session.divider_str, end='\n\n')
    
    def print_system_message(self) -> None:
        print(f'{Color.CYAN}>>> System:{Color.END} {self.instructions}')
        print(f'{Color.CYAN}>>> System: `{self.file_path}`{Color.END}', end='\n\n')
        test_session.print_divider()
    
    def print_chat_message(self) -> None:
        print(f'{Color.CYAN}>>> User:{Color.END} {self.thread_message.content[0].text.value}')
        print(f'{Color.CYAN}>>> Assistant:{Color.END} {self.all_messages.data[0].content[0].text.value}', end='\n\n')
        test_session.print_divider()


if __name__ == '__main__':
    manager_name = input(f'{Color.CYAN}>>> Digite o nome do gerente:{Color.END} ')
    portfolio_id = input(f'{Color.CYAN}>>> Digite o ID da carteira:{Color.END} ')
    print(end='\n\n')
    test_session = Session(
        openai_api_key=OPENAI_API_KEY,
        manager_name=manager_name,
        portfolio_id=portfolio_id,
        current_date=date.today()
    )
    test_session.create_assistant()
    test_session.print_divider()
    test_session.print_system_message()

    while True:
        user_input = input(f'{Color.CYAN}>>> Digite sua pergunta (ou "sair"):{Color.END} ')
        if user_input.lower() == 'sair':
            break
        print(f'\033[1A{Color.CYAN}>>> User:{Color.END} {user_input}\033[K', end='\r')
        test_session.ask(user_input)
        test_session.print_chat_message()
