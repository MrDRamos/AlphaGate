
#https://pythonspot.com/matplotlib-line-chart/

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
 
##############
from pylab import *
t = arange(0.0, 2.0, 0.01)
s = sin(2.5*pi*t)
plot(t, s)
 
xlabel('time (s)')
ylabel('voltage (mV)')
title('Sine Wave')
grid(True)
show()

############
# Matplotlib save figure to image file
y = [2,4,6,8,10,12,14,16,18,20]
x = np.arange(10)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, y, label='$y = numbers')
plt.title('Legend inside')
ax.legend()
#plt.show()
 
fig.savefig('plot.png')

