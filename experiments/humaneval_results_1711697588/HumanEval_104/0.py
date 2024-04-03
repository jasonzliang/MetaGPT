
def unique_digits(x):
    return sorted([i for i in x if all(int(j) % 2 != 0 for j in str(i))])
    