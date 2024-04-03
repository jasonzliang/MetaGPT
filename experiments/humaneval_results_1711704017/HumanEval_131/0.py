
def digits(n):
    """Given a positive integer n, return the product of the odd digits.
    Return 0 if all digits are even.
    For example:
    digits(1)  == 1
    digits(4)  == 0
    digits(235) ==15
    """
    # Convert n to a string so we can iterate over its digits
    str_n = str(n)
    
    # Initialize product at 1, since multiplying by 1 doesn't change the value
    product = 1
    
    # Iterate over each digit in n
    for char in str_n:
        # Convert the character back to an integer
        digit = int(char)
        
        # If the digit is odd, multiply it into our running product
        if digit % 2 == 1:
            product *= digit
    
    # If no odd digits were found, return 0 instead of the initial value of 1
    if product == 1:
        return 0
    else:
        return product
