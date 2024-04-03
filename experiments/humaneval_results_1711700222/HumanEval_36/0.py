
def fizz_buzz(n):
    count = 0
    for i in range(11, n):
        if (i % 11 == 0 or i % 13 == 0) and '7' in str(i):
            count += 1
    return count
