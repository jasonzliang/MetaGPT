
def sum_product(numbers):
    if not numbers:
        return (0, 1)
    else:
        total_sum = sum(numbers)
        product = 1
        for num in numbers:
            product *= num
        return (total_sum, product)
    