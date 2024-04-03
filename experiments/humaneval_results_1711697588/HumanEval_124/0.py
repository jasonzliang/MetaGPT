
import re
from datetime import datetime

def valid_date(date):
    if not date:  # check if the string is empty
        return False
        
    try:
        res = bool(re.match(r'\d{2}-\d{2}-\d{4}', date))  # check if the format is correct
        if not res:
            return False
            
        day, month, year = map(int, date.split('-'))  # split the string into day, month and year
        
        datetime(year, month, day)  # this will throw an error if the date is invalid
    except ValueError:
        return False
    
    return True
