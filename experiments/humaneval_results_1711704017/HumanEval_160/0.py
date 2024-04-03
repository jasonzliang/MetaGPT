
def do_algebra(operator, operand):
    """
    Given two lists operator and operand. The first list has basic algebra operations, and 
    the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.
    """
    
    # Check if inputs are valid
    if not all(isinstance(i, int) for i in operand):
        raise ValueError("All elements in operand list must be integers.")
        
    if len(operator) != len(operand)-1:
        raise ValueError("Length of operator list must be equal to length of operand list minus one.")
    
    # Create a dictionary for operations
    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '//': lambda x, y: x // y,
        '**': lambda x, y: x ** y
    }
    
    # Initialize result with the first operand
    result = operand[0]
    
    # Iterate over operators and operands
    for i in range(len(operator)):
        op = operator[i]
        
        if op not in operations:
            raise ValueError("Invalid operation.")
            
        num = operand[i+1]
        
        # Apply the operation to the result and the next number
        result = operations[op](result, num)
    
    return result
