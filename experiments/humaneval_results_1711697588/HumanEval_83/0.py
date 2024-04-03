def starts_one_ends(n):
    if n == 1:
        return 2
    else:
        return 10**(n-1)*2 - 9*starts_one_ends(n-1)

print(starts_one_ends(3)) 