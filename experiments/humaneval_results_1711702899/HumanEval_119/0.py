
def match_parens(lst):
    """Check if it is possible to concatenate the two strings in some order, 
       that the resulting string will be good.

    Args:
        lst: A list of two strings, both strings consist of open parentheses '(' or close parentheses ')'.

    Returns:
        'Yes' if there's a way to make a good string, and 'No' otherwise.

    Raises:
        TypeError: If the input is not a list of two strings.
        ValueError: If the strings in the list contain characters other than parentheses.
    """
    
    # Check if input is valid
    if not isinstance(lst, list) or len(lst) != 2 or not all(isinstance(i, str) for i in lst):
        raise TypeError("Input should be a list of two strings.")
        
    if any(c not in '()' for s in lst for c in s):
        raise ValueError("Strings can only contain parentheses.")
    
    # Check all possible combinations
    for i in range(2):
        for j in range(2):
            combined = lst[i] + lst[j]
            
            # Count open and close parentheses
            count = 0
            for c in combined:
                if c == '(':
                    count += 1
                else:
                    count -= 1
                    
                # If there's a closing parenthesis without a corresponding opening one, return 'No'
                if count < 0:
                    break
            
            # If all parentheses are balanced, return 'Yes'
            if count == 0:
                return 'Yes'
    
    return 'No'
