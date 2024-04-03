
def simplify(x, n):
    """Simplify the expression x * n. Returns True if x * n evaluates to a whole number and False otherwise."""
    try:
        # Split the fractions into numerator and denominator
        x_num, x_denom = map(int, x.split('/'))
        n_num, n_denom = map(int, n.split('/'))
        
        # Calculate the product of the numerators and the product of the denominators
        num_product = x_num * n_num
        denom_product = x_denom * n_denom
        
        # Check if the result is a whole number
        return num_product % denom_product == 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
