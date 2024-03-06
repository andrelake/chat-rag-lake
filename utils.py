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
    return {'debit': 'débito', 'credit': 'crédito', '': 'benefícios'}[product or '']


def threat_card_variant(card_variant):
    return {'black': 'BLACK', 'gold': 'GOLD', 'platinum': 'PLATINUM', 'standard': 'STANDARD', '': 'INTERNACIONAL'}[card_variant or '']


log = Logger()
