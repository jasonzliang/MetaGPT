
def count_nums(arr):
    return sum(sum(int(digit) for digit in str(abs(n))) > 0 for n in arr)
    