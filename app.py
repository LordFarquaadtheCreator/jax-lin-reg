from timer import timer


@timer
def main(x: list, y: list):
    """
    univariate linear regression model
    outputs the function of the line of best fit -> y = a + bx
    x = input

    params:
        x: jax numpy array of x inputs
        y: jax numpy array of y inputs

    output:
        returns a function fitted to x & y scatter plot
    """

    from get_slope import get_slope
    from get_intercept import get_intercept

    a: float = get_slope(x, y)
    b: float = get_intercept(x, y, a)

    def function(c):
        return a * c + b

    return function
