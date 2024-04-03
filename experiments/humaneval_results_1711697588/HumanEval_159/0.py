
def eat(number, need, remaining):
    total = number + min(need, remaining)
    left = max(0, remaining - need)
    return [total, left]
