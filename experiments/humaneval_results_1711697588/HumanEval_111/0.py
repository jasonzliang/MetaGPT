
def histogram(test):
    from collections import Counter
    letters = test.split()
    count = Counter(letters)
    max_count = max(list(count.values()) + [0])
    return {k: v for k, v in count.items() if v == max_count}
