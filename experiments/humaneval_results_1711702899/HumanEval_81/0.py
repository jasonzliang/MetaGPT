
def numerical_letter_grade(gpas):
    """This function takes a list of GPAs and returns the corresponding letter grades.

    Args:
        gpas: A list of GPAs.

    Returns:
        A list of letter grades corresponding to the input GPAs.

    Raises:
        ValueError: If any GPA is not a number or if it's out of range [0, 4].
    """
    
    # Define grade ranges and their corresponding letters
    grade_ranges = {(4.0, float('inf')): 'A+', (3.7, 4.0): 'A', (3.3, 3.7): 'A-',
                    (3.0, 3.3): 'B+', (2.7, 3.0): 'B', (2.3, 2.7): 'B-',
                    (2.0, 2.3): 'C+', (1.7, 2.0): 'C', (1.3, 1.7): 'C-',
                    (1.0, 1.3): 'D+', (0.7, 1.0): 'D', (0.0, 0.7): 'D-',
                    (-float('inf'), 0.0): 'E'}
    
    # Check if all GPAs are numbers and in the range [0, 4]
    for gpa in gpas:
        if not isinstance(gpa, (int, float)) or gpa < 0 or gpa > 4:
            raise ValueError('All GPAs must be numbers between 0 and 4.')
    
    # Convert each GPA to its corresponding letter grade
    letter_grades = []
    for gpa in gpas:
        for (lower, upper), letter in grade_ranges.items():
            if lower <= gpa < upper:
                letter_grades.append(letter)
                break
    
    return letter_grades
