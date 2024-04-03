
def do_algebra(operator, operand):
    """
    Given two lists operator and operand. The first list has basic algebra operations, 
    and the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.

    Args:
        operator: A list of algebraic operators.
        operand: A list of non-negative integers.

    Returns:
        The result of the evaluated algebraic expression.

    Raises:
        ValueError: If the length of operator list is not equal to the length of 
                    operand list minus one.
    
    """
    if len(operator) != len(operand) - 1:
        raise ValueError("The length of operator list must be equal to the "
                         "length of operand list minus one.")
        
    expression = str(operand[0])
    for i in range(len(operator)):
        expression += ' ' + operator[i] + ' ' + str(operand[i+1])
    
    return eval(expression)
