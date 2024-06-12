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
        returns a function fitted to x & y scatter plot & loss
    """

    from get_slope import get_slope
    from get_intercept import get_intercept
    from loss import mse
    import jax.lax as lax

    if len(x) != len(y):
        raise ValueError("x and y must be of the same length")

    a: float = get_slope(x, y)
    b: float = get_intercept(x, y, a)

    def function(c):
        return a * c + b

    # calculate loss
    def get_pred(_, x_pred):
        return None, function(x_pred)

    _, y_pred = lax.scan(get_pred, None, x)

    return (function, mse(y, y_pred))
