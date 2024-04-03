
def even_odd_count(num):
    """Given an integer, returns a tuple containing the count of even and odd digits in the number.

    Args:
        num (int): The input integer.

    Returns:
        A tuple where the first element is the count of even digits and the second element is the count of odd digits.

    Raises:
        TypeError: If the input is not an integer.
    """
    if not isinstance(num, int):
        raise TypeError("Input must be an integer.")
    
    num = abs(num)  # To handle negative numbers
    even_count = 0
    odd_count = 0

    while num > 0:
        digit = num % 10
        if digit % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
        num //= 10
    
    return (even_count, odd_count)
