'''
#Sigmoid Function
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return s, ds
x = np.arange(-6, 6, 0.01)
sigmoid(x)
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(x, sigmoid(x)[0], color="#307EC7", linewidth=3, label="sigmoid")
ax.plot(x, sigmoid(x)[1], color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()
# tanh

import matplotlib.pyplot as plt
import numpy as np
def tanh(x):
  t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
  dt=1-t**2
  return t,dt
z=np.arange(-6,6,0.01)
tanh(z)[0].size, tanh(z) [1].size
fig, ax = plt.subplots (figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(z,tanh(z) [0], color="#307EC7", linewidth=3, label="tanh")
ax.plot(z,tanh(z) [1], color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()
# relu

import matplotlib.pyplot as plt
import numpy as np
def relu(x):
    r = np.maximum(0, x)
    dr = np.where(x <= 0, 0, 1)
    return r, dr

x = np.arange(-6, 6, 0.01) # Range for the x-axis
relu_values, relu_derivatives = relu(x) # Compute ReLU and its derivative
fig, ax = plt.subplots (figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0)) # Set the x-axis at ya
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#Plotting the ReLU and its derivative
ax.plot(x, relu_values, color="#307EC7", linewidth=3, label="ReLU")
ax.plot(x, relu_derivatives, color="#9621E2", linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
fig.show()
# leaky relu

import matplotlib.pyplot as plt
import numpy as np
# Define the Leaky ReLU function and its derivative
def leaky_relu(x, alpha=0.025):
    r = np.maximum(alpha * x, x)
    dr = np.where(x < 0, alpha, 1)
    return r, dr
#Generate x values
x = np.arange(-6, 6, 0.01)
leaky_relu_values, leaky_relu_derivatives = leaky_relu(x)
#Create the plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0)) # Set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Plot the Leaky ReLU and its derivative
ax.plot(x, leaky_relu_values, color="#307EC7", linewidth=1, label="Leaky ReLU")
ax.plot(x, leaky_relu_derivatives, color="#9621E2", linewidth=1, label="Derivative")
ax.legend(loc="upper left", frameon=False)
# Show the plot
plt.show()
# Prelu

import matplotlib.pyplot as plt
import numpy as np
def prelu(x, alpha=0.25):
    r = np.maximum(alpha * x, x)
    dr = np.where(x < 0, alpha, 1)
    return r, dr
x = np.arange(-6, 6, 0.01) # Range for the x-axis
prelu_values, prelu_derivatives = prelu(x)
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0)) # Set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Plotting the PRELU and its derivative
ax.plot(x, prelu_values, color="#307EC7", linewidth=3, label="PRELU")
ax.plot(x, prelu_derivatives, color="#9621E2", linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
plt.show()
# elu

import matplotlib.pyplot as plt
import numpy as np
def elu(x, alpha=1.0):
    r = np.where(x >= 0, x, alpha * (np.exp(x)-1))
    dr = np.where(x >= 0, 1, alpha * np.exp(x))
    return r, dr
x = np.arange(-6, 6, 0.01) # Range for the x-axis
elu_values, elu_derivatives = elu(x)
fig, ax = plt.subplots (figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0)) # Set the x-axis at y=e
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Plotting the ELU and its derivative
ax.plot(x, elu_values, color="#307EC7", linewidth=3, label="ELU")
ax.plot(x, elu_derivatives, color="#9621E2", linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
plt.show()
#softplus
import matplotlib.pyplot as plt
import numpy as np
def softplus(x):
    r = np.log(1 + np.exp(x))
    dr = 1 / (1 + np.exp(-x))
    return r, dr
x = np.arange(-6, 6, 0.01) # Range for the x-axis
softplus_values, softplus_derivatives = softplus(x)
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0)) #Set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#Plotting the Softplus and its derivative
ax.plot(x, softplus_values, color="#307EC7", linewidth=3, label="Softplus")
ax.plot(x, softplus_derivatives, color="#9621E2", linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
plt.show()