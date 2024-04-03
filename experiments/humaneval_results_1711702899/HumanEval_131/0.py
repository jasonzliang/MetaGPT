
def product_of_odd_digits(n):
    """Given a positive integer n, return the product of the odd digits.
    Return 0 if all digits are even.

    Args:
        n (int): A positive integer.

    Returns:
        int: The product of the odd digits in n. If all digits are even, returns 0.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    product = 1
    has_odd = False
    for digit in str(n):
        if int(digit) % 2 != 0:
            product *= int(digit)
            has_odd = True
            
    return product if has_odd else 0
