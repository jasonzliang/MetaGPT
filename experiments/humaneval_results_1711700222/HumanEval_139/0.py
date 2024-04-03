
def special_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n+1):
            fact = i
            for j in range(i-1, 1, -1):
                fact *= j
            result *= fact
        return result
