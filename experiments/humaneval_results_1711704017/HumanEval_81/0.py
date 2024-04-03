
def numerical_letter_grade(grades):
    """Convert a list of GPAs to letter grades."""
    # Define the grade ranges and their corresponding letters.
    grade_ranges = [
        (4.0, 'A+'),
        (3.7, 'A'),
        (3.3, 'A-'),
        (3.0, 'B+'),
        (2.7, 'B'),
        (2.3, 'B-'),
        (2.0, 'C+'),
        (1.7, 'C'),
        (1.3, 'C-'),
        (1.0, 'D+'),
        (0.7, 'D'),
        (0.0, 'D-'),
    ]
    
    # Convert each GPA to a letter grade using the defined ranges.
    letter_grades = []
    for gpa in grades:
        if gpa < 0 or gpa > 4.0:
            raise ValueError("GPA must be between 0 and 4.0")
        
        # Find the first range where the GPA is greater than or equal to the lower bound of the range.
        for grade_range in grade_ranges:
            if gpa >= grade_range[0]:
                letter_grades.append(grade_range[1])
                break
    
    return letter_grades
