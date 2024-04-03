
def sort_third(l):
    thirds = sorted([x for i, x in enumerate(l) if i % 3 == 0])
    j = 0
    for i in range(len(l)):
        if i % 3 == 0:
            l[i] = thirds[j]
            j += 1
    return l
