def continued_fraction(n_terms=50):
    def fraction_recursive(n):
        if n == 0:
            return 0
        num = (2 * n + 1) * (2 * n - 1)
        return num / (4 + fraction_recursive(n - 1))

    return 2 / (3 + fraction_recursive(n_terms))

# Example usage
result = continued_fraction(50)
print(result)
afa