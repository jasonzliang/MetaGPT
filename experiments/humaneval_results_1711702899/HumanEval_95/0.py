
def check_dict_case(dictionary):
    """
    Given a dictionary, return True if all keys are strings in lower 
    case or all keys are strings in upper case, else return False.
    The function should return False is the given dictionary is empty.
    
    Args:
        dictionary: A dictionary to check its keys' cases.
        
    Returns:
        True if all keys are either lowercase or uppercase, False otherwise.
        
    Raises:
        ValueError: If the provided argument is not a dictionary.
    """
    
    # Check if input is a dictionary
    if not isinstance(dictionary, dict):
        raise ValueError("The provided argument must be a dictionary.")
    
    # Return False for empty dictionaries
    if len(dictionary) == 0:
        return False
        
    # Get the first key and check its case
    first_key = next(iter(dictionary))
    is_lower = first_key.islower()
    
    # Check all keys for their case
    for key in dictionary:
        if (is_lower and not key.islower()) or (not is_lower and not key.isupper()):
            return False
            
    return True
