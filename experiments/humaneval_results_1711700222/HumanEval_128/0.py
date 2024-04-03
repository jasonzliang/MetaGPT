
def prod_signs(arr):
    if not arr:
        return None
    product = 1
    summation = 0
    for num in arr:
        product *= (num > 0) - (num < 0)
        summation += abs(num)
    return product * summation
