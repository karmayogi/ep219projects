# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 00:12:01 2016

@author: Amey P Gaikwad
"""

import matplotlib.pyplot as plt                                
import numpy as np                                                   #import the required
import scipy                                                         #libraries
import math                                                                                                            
                  
def poisson(k,lamb):                                                 #define a function which returns the P(x)
    y=(lamb**k)*np.exp(-1*lamb)/math.factorial(k)                    #where P is poisson 
    return y
t=[]                                                                 #define 
s1=[]                                                                #4 empty arrays for storing 
s2=[]                                                                #the values of x,and the values of P(x)
s3=[]                                                                #with different vaues of lambda   
for i in range(1,31):                                                #fill in the empty arrays defined earlier
    t.append(i)
    s1.append(poisson(i,4))
    s2.append(poisson(i,7))
    s3.append(poisson(i,10))

plt.scatter(t,s1,marker = 'o')                                       #scatter plot for the poisson distribution with lambda=4
plt.scatter(t,s2,marker= 'o' )                                       #scatter plot for the poisson distribution with lambda=7  
plt.scatter(t,s3,marker='o')                                         #scatter plot for the poisson distribution with lambda=10
plt.plot(t,s1,'r', label= ' lambda=4')                               #joining the data points of the scatter plots  
plt.plot(t,s2,'b',label=' lamda=7')                                  #for each of the three
plt.plot(t,s3,'g',label=' lambda=10')                                #poisson scatter plots
plt.grid(True)
plt.xlabel('X')                                                      #labelling the axes                                           
plt.ylabel('P(X)')  
plt.title('Poisson distribution for different values of lambda')     #title of the plot
plt.legend(loc='upper right')                                        #specifying the location of the legend
plt.savefig("test2.png")                                             #saving the plot
plt.show()                                                           #showing the final plot      