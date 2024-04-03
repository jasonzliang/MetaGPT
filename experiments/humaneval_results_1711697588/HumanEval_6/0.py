
def parse_nested_parens(paren_string):
    return [max([len(i) for i in paren.split(')') if i]) for paren in paren_string.split()]
