
def is_prime(n):
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_length(string):
    return is_prime(len(string))
```
```python
print(prime_length('Hello')) # Returns: True
print(prime_length('abcdcba')) # Returns: True
print(prime_length('kittens')) # Returns: True
print(prime_length('orange')) # Returns: False
