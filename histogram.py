import numpy as np
import scipy as sp
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab

################Define stuff###################
e=2.71821
sigma=100

def en(x):
	return 1000*e**(-1*x/10)

def signalf(e, sigma):
	if e>5 and e<15:
		return 20*sigma*(e-5)
	if e>15 and e<25:
		return 20*sigma*(25-e)
	else :
		return 0

################### Reading the data file ####################
df= pd.read_csv("F:/ep219projects/assignment_3_EP219/data.csv")

x=df.E
y=df.Number


############ Plotting the histogram ##########
k= []
for i in range(0,len(x)):
	for j in range(0,y[i]):
		k.append(x[i])
	

"""
plt.hist(k, bins = len(x))
plt.xlabel('Energy')
plt.ylabel('No. Of Events')
plt.title('Total Signal')
#plt.plot(x,y)
plt.show()
"""




################## Expected Number Calculation for each energy value of noise##################
k=[]
lv=0
for i in range(0,len(x)):
	t=int(en(lv+.5))
	#print (t)  #Assuming Value constant for range 0-1, 1-2 etc, I simply need to evaluate it at midpoint and add it to the array.
	k.append(t)
	lv+=1

l= []
for i in range(0,len(x)):
	for j in range(0,k[i]):
		l.append(x[i])

"""
plt.hist(l, bins = len(x))
plt.xlabel('Energy')
plt.ylabel('No. Of Events')
plt.title('Energy vs. No. of Events')

#plt.plot(x,k)
plt.show()
"""

##############Calculating the signal values###########
signaln=[]
lv=0
for i in range(0, len(x)):
	signaln.append(int(signalf(lv+0.5,sigma)))
	lv+=1
	#print (int(signalf(lv+0.5,sigma)))


l= []
for i in range(0,len(x)):
	for j in range(0,signaln[i]):
		l.append(x[i])

"""

plt.hist(l, bins = len(x))
plt.xlabel('Energy')
plt.ylabel('No. Of Events')
plt.title('Signal Energy vs. No. of Events- Sigma= %r' %sigma)
plt.show()
"""


##############Obtaining Theoretical Signal+Background values###########
totalsig=[] 
for i in range(0, len(x)):
	totalsig.append(signaln[i]+k[i])


m= []
for i in range(0,len(x)):
	for j in range(0,totalsig[i]):
		m.append(x[i])
"""
plt.hist(m, bins = len(x))
plt.xlabel('Energy')
plt.ylabel('Total Signal')
plt.title('Signal+Noise vs. Energy - Sigma= %r' %sigma)
plt.show()
"""





"""
#################Finding And Explaining the Log Likelihood Function##########
NOTATION:
di= the ith energy bin frequency that we observed/ obtained from file
ti=Theoretical signal for ith bin (in terms of parameter sigma)
bi= Theoretical background number for the ith bin
Ei= Energy of ith bin
#############
GIVEN:

Signal is:
dN/dE= sigma*20*(E-5)     for    5<Er<15
	  =	sigma*20*(25-E)	for   15<Er<25 

Background is:
dN/dE= 1000exp(-E/10)
############

Assume a Poisson Process for the histogram.

implies P(k)(di)= exp(k)* K^di/di!
here k = ti+bi

Define this to be:= L= Likelihood function (As was done in class)
Implies:
ln(L)= summationOver i[ ti+bi + di*ln(ti+bi)-ln(di!) ]

Ignore bi, ln(di!), as they are constants, will go to zero in differentiation

Now maximise ln(L) wrt sigma

d(ln(L))/d-sigma= summationOver i[20*(Ei-5)+ di*20*(Ei-5)/ln(sigma*20*(Ei-5)+bi)]   
							for 	5<Ei<15

				  summationOver i[20*(25-Ei)+ di*20*(25-Ei)/ln(sigma*20*(25-Ei)+bi)]
							for 	15<Ei<25

Now put this equal to zero to find Max Likelihood Estimate for sigma

"""

###########Actually defining log likelihood############

def loglikelihood(sigma1):
	logsum=0
	var=0
	for i in range(0,len(x)):
		logsum+= signalf(var+0.5,sigma1)+en(var+0.5)+signalf(var+0.5,sigma1)/(sigma1*np.log(signalf(var+0.5,sigma1)))
		var+=1

plt.plot(loglikelihood(sigma), sigma)
plt.show()



