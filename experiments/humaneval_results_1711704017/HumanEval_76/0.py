
def is_simple_power(x, n):
    """Check if a number x is a simple power of n."""
    # Error handling for invalid inputs
    if not isinstance(x, int) or not isinstance(n, int) or x < 1 or n < 2:
        return "Invalid input. Please enter positive integers."
    
    # Calculate the logarithm base n of x and round it to nearest integer
    log_result = round(math.log(x, n))
    
    # Check if n**log_result equals x
    if n ** log_result == x:
        return True
    else:
        return False
