
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def simplify(x, n):
    x_num, x_denom = map(int, x.split('/'))
    n_num, n_denom = map(int, n.split('/'))
    
    product_num = x_num * n_num
    product_denom = x_denom * n_denom
    
    gcd_val = gcd(product_num, product_denom)
    
    return (product_num // gcd_val) % (product_denom // gcd_val) == 0
