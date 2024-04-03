def monotonic(l):
    return (all(x<y for x, y in zip(l, l[1:])) or
            all(x>y for x, y in zip(l, l[1:]))) 