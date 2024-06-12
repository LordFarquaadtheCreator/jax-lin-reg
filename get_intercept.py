from timer import timer


@timer
def get_intercept(x, y, b):
    import jax.numpy as jnp

    """a = y' - bx' linear regression"""
    mean_x: float = jnp.mean(x)
    mean_y: float = jnp.mean(y)
    a: float = mean_y - (b * mean_x)

    return a
