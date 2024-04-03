
def even_odd_count(num):
    num = str(abs(num))
    even = 0
    odd = 0
    for digit in num:
        if int(digit) % 2 == 0:
            even += 1
        else:
            odd += 1
    return (even, odd)
