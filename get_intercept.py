import jax.numpy as jnp
from jax import grad, jit
from get_slope import get_slope


def get_intercept(x, y):
	'''a = y' - bx' linear regression'''
	mean_x: float = jnp.mean(x)
	mean_y: float = jnp.mean(y)

	# formula for the intercept (a)
	# a = mean_y - b * mean_x
	b: float = get_slope(x, y)
	a: float = mean_y - (b * mean_x)
    
	return a