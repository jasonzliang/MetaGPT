
def get_odd_collatz(n):
    collatz = []
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        if n not in collatz:
            collatz.append(n)
    return sorted([i for i in collatz if i%2 != 0])
