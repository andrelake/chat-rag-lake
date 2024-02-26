class Logger:
    def __init__(self, end: str = '\n', verbose: bool = False, **kwargs):
        self.end = end
        self.verbose = verbose
        self.kwargs = kwargs

    def __call__(self, message: str, **kwargs):
        kwargs = {'end': self.end, **self.kwargs, **kwargs}
        if self.verbose:
            print(message, **kwargs)


def get_month_name(n: int) -> str:
    return ('janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro')[n-1]


def threat_product(product):
    return {'debit': 'débito', 'credit': 'crédito', '': 'desconhecido'}[product or '']


log = Logger()
