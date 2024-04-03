
def solve(N):
    """Given a positive integer N, return the total sum of its digits in binary.

    Args:
        N (int): A positive integer between 0 and 10000.

    Returns:
        str: The binary representation of the sum of the digits in N.
    """
    # Check if input is valid
    if not isinstance(N, int) or N < 0 or N > 10000:
        raise ValueError("Input must be a positive integer between 0 and 10000.")
    
    # Calculate the sum of digits in binary
    digit_sum = sum([int(digit) for digit in str(N)])
    
    return bin(digit_sum)[2:]
