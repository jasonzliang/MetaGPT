
def sort_even(l):
    even_indices = sorted([v for i, v in enumerate(l) if i % 2 == 0])
    result = [None] * len(l)
    result[::2] = even_indices
    result[1::2] = l[1::2]
    return result
