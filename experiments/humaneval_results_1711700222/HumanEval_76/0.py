
def is_simple_power(x, n):
    if x < 1 or n < 2:
        return False
    while x % n == 0:
        x /= n
    return x == 1
