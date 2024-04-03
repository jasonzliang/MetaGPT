
def search(lst):
    from collections import Counter
    counter = Counter(lst)
    for num in sorted(counter.keys(), reverse=True):
        if num <= counter[num]:
            return num
    return -1
    