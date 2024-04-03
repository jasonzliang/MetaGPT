
import re
from datetime import datetime

def valid_date(date):
    if not date:  # check if the string is empty
        return False
        
    try:
        res = bool(datetime.strptime(date, '%m-%d-%Y'))  # check if the format is correct
    except ValueError:
        res = False
    
    if res:
        day, month, year = map(int, date.split('-'))  # split the string into day, month and year
        
        # check if the months are valid
        if not (1 <= month <= 12):  
            return False
            
        # check if the days are valid for each month
        if month in [4,6,9,11] and not(1 <= day <= 30): 
            return False
        elif month == 2:
            if year % 4 == 0:  # check for leap years
                if not (1 <= day <= 29):  
                    return False
            else:
                if not (1 <= day <= 28):  
                    return False
        else:
            if not(1 <= day <= 31):  
                return False
    return res
