
def fibfib(n):
    if n < 0:
        return "Input should be non-negative"
    elif n == 0 or n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b, c = 0, 0, 1
        for _ in range(3, n + 1):
            a, b, c = b, c, a + b + c
        return c
