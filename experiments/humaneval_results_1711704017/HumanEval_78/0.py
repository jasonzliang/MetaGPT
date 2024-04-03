
def hex_key(num):
    """Count the number of hexadecimal digits that are primes."""
    # Define a list of prime numbers in hexadecimal
    primes = ['2', '3', '5', '7', 'B', 'D']
    
    # Initialize counter for prime digits
    count = 0
    
    # Iterate over each character in the input string
    for digit in num:
        # If the character is a prime, increment the counter
        if digit.upper() in primes:
            count += 1
            
    return count
