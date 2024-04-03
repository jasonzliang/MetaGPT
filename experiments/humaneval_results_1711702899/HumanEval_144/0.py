
def simplify(x, n):
    """This function checks if the product of two fractions is a whole number.

    Args:
        x (str): The first fraction in the format 'numerator/denominator'.
        n (str): The second fraction in the same format as x.

    Returns:
        bool: True if the product of x and n is a whole number, False otherwise.

    Raises:
        ValueError: If either x or n are not valid fractions.
    """
    try:
        # Split the string into numerator and denominator
        x_num, x_denom = map(int, x.split('/'))
        n_num, n_denom = map(int, n.split('/'))
        
        # Calculate the product of the fractions
        product = (x_num * n_num) / (x_denom * n_denom)
        
        # Check if the product is a whole number
        return product == int(product)
    except ValueError:
        raise ValueError("Both x and n must be valid fractions in the format 'numerator/denominator'")
