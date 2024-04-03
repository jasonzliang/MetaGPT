
from typing import List

def parse_nested_parens(paren_string: str) -> List[int]:
    """This function takes a string of multiple groups for nested parentheses separated by spaces 
    and returns the deepest level of nesting for each group.

    Args:
        paren_string: A string represented multiple groups for nested parentheses separated by spaces.

    Returns:
        A list of integers where each integer represents the maximum depth of nesting in a group.

    Raises:
        ValueError: If the input string contains any character other than '(' and ')'.
    """
    
    # Check if the string only contains parentheses
    for char in paren_string:
        if char != "(" and char != ")":
            raise ValueError("Input string should only contain parentheses.")
            
    max_depths = []
    groups = paren_string.split()  # Split the string into groups
    
    for group in groups:
        current_depth = 0
        max_depth = 0
        
        for char in group:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1
                
        max_depths.append(max_depth)
        
    return max_depths
