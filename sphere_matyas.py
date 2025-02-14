from dezero import Variable
import numpy as np

def sphere(x,y):
    z=x**2+y**2
    return z

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=sphere(x,y)
z.backward()
print(z)
print(x.grad, y.grad)

def matyas(x,y):
    z=0.26*(x**2+y**2)-0.48*x*y
    return z
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=matyas(x,y)
z.backward()
print(z)
print(x.grad, y.grad)