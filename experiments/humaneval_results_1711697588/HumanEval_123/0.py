
def get_odd_collatz(n):
    collatz = []
    while n != 1:
        collatz.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    collatz.append(1)
    
    return sorted([num for num in set(collatz) if num % 2 != 0])
