
def is_sorted(lst):
    return lst == sorted(lst) and len(lst) == len(set(lst))
    