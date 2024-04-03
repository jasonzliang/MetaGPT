def fib(n):
    if n <= 0:
        return "Input should be positive integer."
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a = 0
        b = 1
        for i in range(2, n):
            fib = a + b
            a, b = b, fib
        return b

print(fib(10)) # Outputs: 34
print(fib(1)) # Outputs: 0
print(fib(8)) # Outputs: 13