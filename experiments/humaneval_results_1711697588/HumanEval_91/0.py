
def is_bored(S):
    sentences = S.replace('?', '.').replace('!', '.').split('.')
    return sum([1 for sentence in sentences if sentence.strip().startswith("I")])
