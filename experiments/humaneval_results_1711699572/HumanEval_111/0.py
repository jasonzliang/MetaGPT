
def histogram(test):
    from collections import Counter
    letters = test.split()
    if not letters:
        return {}
    count = Counter(letters)
    max_count = max(list(count.values()))
    result = {k: v for k, v in count.items() if v == max_count}
    return result
