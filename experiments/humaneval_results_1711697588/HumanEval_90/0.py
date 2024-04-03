
def next_smallest(lst):
    if len(set(lst)) <= 1:
        return None
    else:
        lst = sorted(list(set(lst)))
        return lst[1]
    