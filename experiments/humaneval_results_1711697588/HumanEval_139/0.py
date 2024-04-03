def special_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(1, n+1):
            temp = i
            while temp > 0:
                result *= temp
                temp -= 1
        return result

print(special_factorial(4)) 