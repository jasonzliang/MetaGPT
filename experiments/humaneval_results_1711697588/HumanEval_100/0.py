
def make_a_pile(n):
    pile = [n]
    while n > 0:
        if n % 2 == 0:
            n += 1
        else:
            n += 2
        pile.append(n)
    return pile[1:]
