
def right_angle_triangle(a, b, c):
    sides = sorted([a, b, c])
    return sides[2]**2 == sides[0]**2 + sides[1]**2
