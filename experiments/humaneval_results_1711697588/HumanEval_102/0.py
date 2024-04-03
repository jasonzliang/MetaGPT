def choose_num(x, y):
    for i in range(y, x-1, -1):
        if i % 2 == 0:
            return i
    return -1

print(choose_num(12, 15)) # Outputs: 14
print(choose_num(13, 12)) # Outputs: -1 