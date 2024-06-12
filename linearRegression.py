import jax.numpy as jnp
from jax import grad, jit

## a = y' - bx' linear regression

## dummy data
x = jnp.array([1,2,3,4,5])
y = jnp.array([2,4,6,8,10])

##
# calculate the mean of both variables??
mean_x = jnp.mean(x)
mean_y = jnp.mean(y)

print(mean_x)
print(mean_y)
# formula for slope (b)??????

# formula for the intercept (a)
#a = mean_y - b * mean_x