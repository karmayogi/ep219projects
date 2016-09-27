# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:57:04 2016

@author: Amey P Gaikwad
"""
import matplotlib.pyplot as plt                                                               #importing the
import numpy as np                                                                            #required 
from matplotlib.font_manager import FontProperties                                            #libraries 

def gauss(x,u,s):                                                                             #defining a function that returns the value of P(x)
    y=np.exp(-1*(x-u)*(x-u)/(2*s))/np.sqrt(2*np.pi*s)                                         #for a given value of x where P(x)
    return y                                                                                  #is the Gaussian distribution

t=np.arange(-30.0,30.0,0.01)                                                                  #sets the range of values of x
s1=gauss(t,-5,16)                                                                             #gaussian with mean=-5,sd=4
s2=gauss(t,0,49)                                                                              #gaussian with mean=0,sd=7
s3=gauss(t,5,25)                                                                              #gaussian with mean=5,sd=5  
plt.plot(t,s1,'r',label='mean=-5;sd=4')                                                       #plots the first gaussian  
plt.plot(t,s2,'b',label='mean=0;sd=7')                                                        #plots the second gaussian 
plt.plot(t,s3,'g',label='mean=5;sd=5')                                                        #plots the third gaussian
plt.xlabel('x')                                                                               #labels the x axis 
plt.ylabel('y')                                                                               #labels the y axis
plt.title('Gaussian distributions\n for different values of the mean and standard deviation') # gives title to the plot
fontP = FontProperties()                                  
fontP.set_size('small')                                                                       #set font size to small
plt.grid(True)  
plt.legend(loc='upper right',shadow='true',prop = fontP)                                      #legend
plt.savefig("gauss.png")                                                                      #saves the figure
plt.show()                                                                                    #shows the plot