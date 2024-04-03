
def solution(lst):
    return sum([val for i, val in enumerate(lst) if i % 2 == 0 and val % 2 != 0])
    