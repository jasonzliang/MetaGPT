
import re
from datetime import datetime

def valid_date(date):
    """Validate a given date string and returns True if the date is valid otherwise False.
    The date is valid if all of the following rules are satisfied:
    1. The date string is not empty.
    2. The number of days is not less than 1 or higher than 31 days for months 1,3,5,7,8,10,12. And the number of days is not less than 1 or higher than 30 days for months 4,6,9,11. And, the number of days is not less than 1 or higher than 29 for the month 2.
    3. The months should not be less than 1 or higher than 12.
    4. The date should be in the format: mm-dd-yyyy"""
    if not date:
        return False
    try:
        # Check if date matches the required format
        if re.match('^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\\d{4}$', date):
            day, month, year = map(int, date.split('-'))
            # Check if the date is valid
            datetime(year, month, day)
            return True
        else:
            return False
    except ValueError:
        return False
