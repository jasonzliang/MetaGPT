
def below_zero(operations):
    """Check if the balance of a bank account falls below zero."""
    balance = 0  # Start with zero balance.
    for operation in operations:
        balance += operation  # Apply each operation to the balance.
        if balance < 0:  # If the balance is now negative, return True.
            return True
    return False  # If no operation caused the balance to fall below zero, return False.
