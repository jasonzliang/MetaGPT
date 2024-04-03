
def decimal_to_binary(decimal):
    """Convert a given decimal number to binary format."""
    if not isinstance(decimal, int) or decimal < 0:
        return "Error: Input must be a non-negative integer."
    elif decimal == 0:
        return 'db0db'
    else:
        binary = ''
        while decimal > 0:
            binary = str(decimal % 2) + binary
            decimal //= 2
        return 'db' + binary + 'db'
