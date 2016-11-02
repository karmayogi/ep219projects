import numpy as np
import scipy as sp
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab

################Define stuff###################
e=2.71821
sigma=1

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




################## Expected Number Calculation for each energy value of NOISE##################
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
	#var=0
	for i in range(0,len(x)):
		logsum +=  y[i]*np.log(signalf(i+0.5,sigma1)+en(i+0.5))-signalf(i+0.5,sigma1)-en(i+0.5)
		#print (logsum)
		#var+=1
	return logsum

var=18
sigma1=1
i=10
#print (totalsig[i]+y[i]*signalf(var+0.5,sigma1)/(sigma1*np.log(totalsig[i])))
#print (loglikelihood(sigma1))


logsig = []
z=[]
var=0.001
for i in range(0,100000,1):
	logsig.append(loglikelihood(i*var))
	z.append(i*var)


imax=np.argmax(logsig)	
print (imax, z[imax],logsig[imax])

##Since I know that sigma corresponding to 1 sigma interval lies close to 1, I need to look only in that interval.
## Also i need to take in some tolerance to be abe to pinpoint the value of sigma of the required error interval


#####Finding the sigma corresponding to 1 sigma error interval
for i in range(0, 500):
	if (abs(logsig[imax]-0.5-logsig[i])<0.03):
		print (z[i])


plt.plot(z, logsig)
plt.xlabel('Sigma')
plt.ylabel('Log Likelihood')
plt.title('Plot of Log Likelihood vs. Sigma with MLE= 0.177')
plt.show()





## Max likelihood estimate comes out to be 0.177 
## Lower limit of 1 sigma interval comes out 0.147
## Upper limit of 1 sigma interval comes out 0.206

"""
Final conclusion:
Yes the data favours the presence of a dark matter signal
Reasons:
The cross section is within reasonable limits and so is the MLE along with the 1 sigma error interval
"""




