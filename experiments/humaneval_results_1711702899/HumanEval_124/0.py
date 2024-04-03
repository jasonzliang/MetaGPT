
def valid_date(date):
    """This function checks if a given date string is valid or not.

    Args:
        date (str): The date string to be checked in the format 'mm-dd-yyyy'.

    Returns:
        bool: True if the date is valid, False otherwise.

    Raises:
        ValueError: If the input is not a string or does not match the required format.
    """
    
    # Check if the input is a string
    if not isinstance(date, str):
        raise ValueError("The input must be a string.")
        
    # Split the date into month, day and year
    try:
        month, day, year = map(int, date.split('-'))
    except ValueError:
        return False
    
    # Check if the date is in correct format (mm-dd-yyyy)
    if len(date.split('-')) != 3:
        return False
        
    # Check if month, day and year are within valid range
    if not 1 <= month <= 12 or not 1 <= day <= 31 or year < 0:
        return False
    
    # Check for February with leap years
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            if day > 29:
                return False
        elif day > 28:
            return False
    
    # Check for months with 30 days
    if month in [4, 6, 9, 11] and day > 30:
        return False
        
    return True
