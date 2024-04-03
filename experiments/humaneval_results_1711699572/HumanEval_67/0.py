
def fruit_distribution(s,n):
    fruits = s.split(" and ")
    apples = int(fruits[0].split()[0])
    oranges = int(fruits[1].split()[0])
    return n - apples - oranges
