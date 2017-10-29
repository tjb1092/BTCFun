from matplotlib.ticker import FormatStrFormatter
from random import randint
import random as rand
import matplotlib.pyplot as plt
import numpy as np


##################################                      #k-NN calculations
def KNN(testSet, kVal):
	Q = [0,0,0,0]
	KError = 0
	for i in range(0,len(testSet)):                            
		Dvals = []
		for j in range(0,kVal): Dvals.insert(len(Dvals),D[radii[testSet[i]].index(sortRad[testSet[i]][j+1])])
		dvals = np.array(Dvals)
		Q = Qcounter(Q,int(round(np.average(dvals))),D[testSet[i]])	
	return Q

###################################                     #neighborhood-based classifier
def neighbor(testSet, radius):
	Q = [0,0,0,0]
	RError = 0
	for i in range(0,len(testSet)):                            
		Dvals = []
		j = 1
		while  sortRad[testSet[i]][j] <= radius and j+1 < len(L):
			Dvals.insert(len(Dvals),D[radii[testSet[i]].index(sortRad[testSet[i]][j])])
			j = j + 1
		if len(Dvals) == 0: Dvals.insert(len(Dvals),0)
		dvals = np.array(Dvals)
		Q = Qcounter(Q,int(round(np.average(dvals))),D[testSet[i]])
	return Q

#######################################                #Perceptron algorithm
def prediction(Lpt, Ppt, Win):              #Predicts Perceptron output
	actvFunc = Lpt*Win[0]+Ppt*Win[1]+Win[2]
	return 1.0 if actvFunc >= 0.0 else 0.0

def train(trainIndex, w, trate, n):
	Q = [0,0,0,0]
	errorRate =  float(D.count(0))/float(len(D))
	for i in range(0,n):
		totalerror = 0.0
		for j in range(0,len(trainIndex)):
			predicted = prediction(L[trainIndex[j]],P[trainIndex[j]],w)
			error = D[trainIndex[j]] - predicted
			Q = Qcounter(Q,predicted,D[trainIndex[j]])
			w[0] = w[0]+L[trainIndex[j]]*trate*error
			w[1] = w[1]+P[trainIndex[j]]*trate*error
			w[2] = w[2]+trate*error
			totalerror = totalerror + (error)**2
		errorRate = totalerror/int(round(0.8*len(L)))
	return Q,(1-errorRate),w

def perceptron(testIndex, trainIndex, Wout):
	#testing
	trainHR = []; testHR = []
	trainQ = [0,0,0,0]; testQ = [0,0,0,0]
	TeParams = [0,0,0,0,0]; TrParams = [0,0,0,0,0]
	trainingRate = 0.1

	for ep in range(0,11):
		trainHR.insert(len(trainHR),0)
		trainQ,trainHR[ep],Wout = train(trainIndex, Wout,trainingRate,10*(ep)) 
		Terrors = 0 
		for t in range(0,len(testIndex)):        #test weights
			predicted = prediction(L[testIndex[t]],P[testIndex[t]],Wout)
			testQ =  Qcounter(testQ,predicted,D[testIndex[t]])
			if predicted != D[testIndex[t]]: Terrors += 1
		testHR.insert(len(testHR),(1-Terrors/(int(round(0.2*len(L))))))
	TrParams[0],TrParams[1],TrParams[2],TrParams[3],TrParams[4] = ParameterCalc(trainQ)
	TeParams[0],TeParams[1],TeParams[2],TeParams[3],TeParams[4] = ParameterCalc(testQ)
	return TeParams, TrParams, testHR, trainHR


#################################################################
def Qcounter(Q, Predict, Actual):
	if Predict > Actual: Q[1] += 1
	if Predict < Actual: Q[2] += 1
	if Predict == Actual and Actual == 0: Q[0] += 1
	if Predict == Actual and Actual == 1: Q[3] += 1
	return Q

def ParameterCalc(Q):
	if Q[0]+Q[3] == 0: HR = 0
	else: HR = (Q[0]+Q[3])/(Q[0]+Q[1]+Q[2]+Q[3])
	if Q[3] == 0:
		Sens = 0
		PPV = 0
	else: 
		Sens = Q[3]/(Q[3]+Q[2])
		PPV = Q[3]/(Q[3]+Q[1])
	if Q[0] == 0: 
		Spec = 0
		NPV = 0
	else: 
		Spec = Q[0]/(Q[0]+Q[1])
		NPV = Q[0]/(Q[0]+Q[2])
	return HR, Sens, Spec, PPV, NPV

def autolabel(rects,ax,heights):
    """
    Attach a text label above each bar displaying its height
    """
    i = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2., 1.00*heights[i], '%.2f' % heights[i], ha='center', va='bottom')
        i += 1
################################################################################################

################################################################################################

################################################################################################

data_file = open("hw2_dataProblem.txt", "r")            #This file has the disease data
lines = data_file.read().split('\n')                    #Split file by new line character

L = []
P = []
D = []
originalL = []
originalP = []

for i in range(2, len(lines)):                          #Parses lines of text file to create L,P,D arrays
	values = lines[i].split(' ')
	if len(values) > 3:
		values = [val for val in values if val.strip()] #removes array values that have a space
		originalL.insert(len(originalL),float(values[0]))
		L.insert(len(L),float(values[0]))
		originalP.insert(len(originalP),float(values[1]))
		P.insert(len(P),float(values[1]))
		D.insert(len(D),int(values[2]))

maxVal = max(max(L),max(P))
minVal = min(min(L), min(P))

L[:] = [(x - minVal)/(maxVal-minVal) for x in L] #Normalizes Data to have a min of 0 and max of 1
P[:] = [(x - minVal)/(maxVal-minVal) for x in P]

radii = [[0 for x in range(len(L))] for y in range(len(L))] 
sortRad = [[0 for x in range(len(L))] for y in range(len(L))] 
for i in range(0,len(L)):                               #Pick point to find radii of
	testL = L[i]
	testP = P[i]

	for j in range(0,len(L)):
		radii[i][j] = pow(((testL-L[j])**2)+((testP-P[j])**2),0.5)
	sortRad[i] = sorted(radii[i])

trials = np.array([1,2,3,4,5,6,7,8,9])
testIndx = [[0 for x in range(int(0.2*len(L)))] for y in range(9)] 
trainIndx = [[0 for x in range(len(L)-int(0.2*len(L)))] for y in range(9)] 

for trl in range(9):
	indx = 0
	for i in range(0,int(0.2*len(L))): 
		newVal = randint(0,len(L)-1)
		while testIndx[trl].count(newVal) > 0:
			newVal = randint(0,len(L)-1)
		testIndx[trl][i] = newVal
	for i in range(0,len(L)-1):
		if testIndx[trl].count(i) == 0:
			trainIndx[trl][indx] = i
			indx += 1

############################################################
k = 9
HR = []; Sens = []; Spec = []; PPV = []; NPV = []
for trl in range(9):
	KQ = KNN(trainIndx[trl], k)
	HR.insert(len(HR),0)
	Sens.insert(len(Sens),0)
	Spec.insert(len(Spec),0)
	PPV.insert(len(PPV),0)
	NPV.insert(len(NPV),0)
	HR[trl], Sens[trl], Spec[trl], PPV[trl], NPV[trl] = ParameterCalc(KQ)
print(np.mean(np.array(HR)),' ',np.mean(np.array(Sens)),' ',np.mean(np.array(Spec)),' ',np.mean(np.array(PPV)),' ',np.mean(np.array(NPV)))
print(np.std(np.array(HR)),' ',np.std(np.array(Sens)),' ',np.std(np.array(Spec)),' ',np.std(np.array(PPV)),' ',np.std(np.array(NPV)))
#################Best K Plot
plt.figure(1)
trainSetIndx = trainIndx[HR.index(max(HR))]
for i in range(0,len(trainSetIndx)):                            
	Dvals = []
	for j in range(0,k): Dvals.insert(len(Dvals),D[radii[trainSetIndx[i]].index(sortRad[trainSetIndx[i]][j+1])])
	dvals = np.array(Dvals)
	if(int(round(np.average(dvals)))) == 1: 
		plt.scatter(L[trainSetIndx[i]],P[trainSetIndx[i]],s=20,c='r')
	else: plt.scatter(L[trainSetIndx[i]],P[trainSetIndx[i]],s=20,c='b')
for m in range(0,51):
	for i in range(0,51):                               #Pick point to find radii of
		radi = []
		dgrid =[]
		testL = m/50
		testP = i/50
		for j in range(0,len(L)):
			radi.insert(len(radi),pow(((testL-L[j])**2)+((testP-P[j])**2),0.5))
		sortradi = sorted(radi)
		for j in range(0,k):
			dgrid.insert(len(dgrid),D[radi.index(sortradi[j])])
		if len(dgrid) == 0: dgrid.insert(len(dgrid),0)
		dvals = np.array(dgrid)
		if(int(round(np.average(dvals)))) == 1: 
			plt.scatter(testL,testP,s=5,c='r')
		else: plt.scatter(testL,testP,s=5,c='b')
plt.title('Best k-NN Classification Boundary')
plt.ylabel('Normalized Blood Pressure Levels')
plt.xlabel('Normalized Sodium Levels') 
plt.show()

plt.figure(1)
plt.suptitle('k-NN Classifier Output Parameters')
K1 = plt.subplot(221)
rectsK1 = plt.bar(trials, PPV, 0.75, color="blue")
autolabel(rectsK1,K1,PPV)
plt.ylim([0,1.05])
plt.title('PPV Per Trial')
plt.ylabel('PPV')
plt.xlabel('Trial') 
K2 = plt.subplot(222)
rectsK2 = plt.bar(trials, NPV, 0.75, color="blue")
autolabel(rectsK2,K2,NPV)
plt.ylim([0,1.05])
plt.title('NPV Per Trial')
plt.ylabel('NPV')
plt.xlabel('Trial') 
K3 = plt.subplot(223)
rectsK3 = plt.bar(trials, Sens, 0.75, color="blue")
autolabel(rectsK3,K3,Sens)
plt.ylim([0,1.05])
plt.title('Sensitivity Per Trial')
plt.ylabel('Sensitivity')
plt.xlabel('Trial') 
K4 = plt.subplot(224)
rectsK4 = plt.bar(trials, Spec, 0.75, color="blue")
autolabel(rectsK4,K4,Spec)
plt.ylim([0,1.05])
plt.title('Specificity Per Trial')
plt.ylabel('Specificity')
plt.xlabel('Trial')  
plt.show()
############################################################
Rad = 0.054

RHR = []; RSens = []; RSpec = []; RPPV = []; RNPV = []
for trl in range(9):
	RQ = neighbor(trainIndx[trl], Rad)
	RHR.insert(len(HR),0)
	RSens.insert(len(Sens),0)
	RSpec.insert(len(Spec),0)
	RPPV.insert(len(PPV),0)
	RNPV.insert(len(NPV),0)
	RHR[trl], RSens[trl], RSpec[trl], RPPV[trl], RNPV[trl] = ParameterCalc(RQ)
print(np.mean(np.array(RHR)),' ',np.mean(np.array(RSens)),' ',np.mean(np.array(RSpec)),' ',np.mean(np.array(RPPV)),' ',np.mean(np.array(RNPV)))
print(np.std(np.array(RHR)),' ',np.std(np.array(RSens)),' ',np.std(np.array(RSpec)),' ',np.std(np.array(RPPV)),' ',np.std(np.array(RNPV)))
#################Best R Plot
plt.figure(1)
trainSetIndx = trainIndx[RHR.index(max(RHR))]
for i in range(0,len(trainSetIndx)):                          
	Dvals = []
	j = 0
	while  sortRad[trainSetIndx[i]][j] <= Rad and j+1 < len(L):
		Dvals.insert(len(Dvals),D[radii[trainSetIndx[i]].index(sortRad[trainSetIndx[i]][j])])
		j = j + 1
	if len(Dvals) == 0: Dvals.insert(len(Dvals),0)
	dvals = np.array(Dvals)
	if(int(round(np.average(dvals)))) == 1: 
		plt.scatter(L[trainSetIndx[i]],P[trainSetIndx[i]],s=20,c='r')
	else: plt.scatter(L[trainSetIndx[i]],P[trainSetIndx[i]],s=20,c='b')
for m in range(0,51):
	for i in range(0,51):                               #Pick point to find radii of
		radi = []
		dgrid =[]
		testL = m/50
		testP = i/50
		for j in range(0,len(L)):
			radi.insert(len(radi),pow(((testL-L[j])**2)+((testP-P[j])**2),0.5))
		sortradi = sorted(radi)
		n = 0
		while  sortradi[n] <= Rad and n+1 < len(L):
			dgrid.insert(len(dgrid),D[radi.index(sortradi[n])])
			n = n + 1
		if len(dgrid) == 0: dgrid.insert(len(dgrid),0)
		dvals = np.array(dgrid)
		if(int(round(np.average(dvals)))) == 1: 
			plt.scatter(testL,testP,s=5,c='r')
		else: plt.scatter(testL,testP,s=5,c='b')

plt.title('Best Neighborhood-Based Classification Boundary')
plt.ylabel('Normalized Blood Pressure Levels')
plt.xlabel('Normalized Sodium Levels') 
plt.show()

plt.figure(2)
plt.suptitle('Neighborhood-Based Classifier Output Parameters')
R1 = plt.subplot(221)
rectsR1 = plt.bar(trials, RPPV, 0.75, color="blue")
autolabel(rectsR1,R1,RPPV)
plt.ylim([0,1.05])
plt.title('PPV Per Trial')
plt.ylabel('PPV')
plt.xlabel('Trial') 
R2 = plt.subplot(222)
rectsR2 = plt.bar(trials, RNPV, 0.75, color="blue")
autolabel(rectsR2,R2,RNPV)
plt.ylim([0,1.05])
plt.title('NPV Per Trial')
plt.ylabel('NPV')
plt.xlabel('Trial') 
R3 = plt.subplot(223)
rectsR3 = plt.bar(trials, RSens, 0.75, color="blue")
autolabel(rectsR3,R3,RSens)
plt.ylim([0,1.05])
plt.title('Sensitivity Per Trial')
plt.ylabel('Sensitivity')
plt.xlabel('Trial') 
R4 = plt.subplot(224)
rectsR4 = plt.bar(trials, RSpec, 0.75, color="blue")
autolabel(rectsR4,R4,RSpec)
plt.ylim([0,1.05])
plt.title('Specificity Per Trial')
plt.ylabel('Specificity')
plt.xlabel('Trial')  
plt.show()

############################################################
weights = [0.0, 0.0, 0.0]  #[weight of L, weight of P, weight of bias]
for j in range(0,3): 
	weights[j] = rand.uniform(-5.0, 5.0)

TestParams = [[0 for x in range(5)] for y in range(9)] # HR, Sens, Spec, PPV, NPV
TrainParams = [[0 for x in range(5)] for y in range(9)] # HR, Sens, Spec, PPV, NPV
testHR = [[0 for x in range(4)] for y in range(9)]
trainHR = [[0 for x in range(4)] for y in range(9)]

for i in range(9):
	weights = [0.0, 0.0, 0.0]  #[weight of L, weight of P, weight of bias]
	TestParams[i], TrainParams[i], testHR[i], trainHR[i] = perceptron(testIndx[i], trainIndx[i], weights)
TEperms = np.array(TestParams)
TRperms = np.array(TrainParams)
TRerror= 1-np.array(trainHR)
TEerror= 1-np.array(testHR)
print(np.mean(np.array(TestParams[0])),' ',np.mean(np.array(TestParams[1])),' ',np.mean(np.array(TestParams[2])),' ',np.mean(np.array(TestParams[3])),' ',np.mean(np.array(TestParams[4])))
print(np.std(np.array(TestParams[0])),' ',np.std(np.array(TestParams[1])),' ',np.std(np.array(TestParams[2])),' ',np.std(np.array(TestParams[3])),' ',np.std(np.array(TestParams[4])))

plt.figure(3)
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[0])
plt.plot(10*np.array(range(0,len(testHR[0]))), TEerror[0])
plt.title('Perceptron Error Rates')
plt.ylabel('Error Rate')
plt.xlabel('Epochs')
plt.ylim(0,1)
plt.show()
#####################TRIAL WISE GRAPH
plt.figure(3)
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[0])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[1])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[2])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[3])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[4])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[5])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[6])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[7])
plt.plot(10*np.array(range(0,len(trainHR[0]))), TRerror[8])
plt.axis([0, 100, 0.1, 0.31])
plt.title('Perceptron Error Rate Over 9 Trials')
plt.ylabel('Error Rate')
plt.xlabel('Epochs')
plt.show()
###################Mean Training Error
means = []
stdevs = []
plt.figure(4)
for i in range(0,11):
	means.insert(len(means),np.mean(TRerror[:,i]))
	stdevs.insert(len(stdevs),np.std(TRerror[:,i]))
	plt.plot([i*10,i*10],[(means[i]+stdevs[i]),(means[i]-stdevs[i])], color="black")
plt.plot(np.array(range(0,11))*10,means)
plt.title('Perceptron Average Error Rate for 9 Trials')
plt.ylabel('Average Error Rate')
plt.xlabel('Epochs')
plt.show()


width = 0.375
plt.figure(3)
plt.suptitle('Perceptron Output Parameters')
P1 = plt.subplot(221)
rectsP1 = plt.bar(trials-width/2, TRperms[:,3], width, color="blue", label = 'Training')
autolabel(rectsP1,P1,TRperms[:,3])
rectsP1 = plt.bar(trials+width/2, TEperms[:,3], width, color="orange", label = 'Test')
autolabel(rectsP1,P1,TEperms[:,3])
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.ylim([0,1.05])
plt.title('PPV Per Trial')
plt.ylabel('PPV')
plt.xlabel('Trial') 
P2 = plt.subplot(222)
rectsP2 = plt.bar(trials-width/2, TRperms[:,4], width, color="blue", label = 'Training')
autolabel(rectsP2,P2,TRperms[:,4])
rectsP2 =plt.bar(trials+width/2, TEperms[:,4], width, color="orange", label = 'Test')
autolabel(rectsP2,P2,TEperms[:,4])
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.ylim([0,1.05])
plt.title('NPV Per Trial')
plt.ylabel('NPV')
plt.xlabel('Trial') 
P3 = plt.subplot(223)
rectsP3 = plt.bar(trials-width/2, TRperms[:,1], width, color="blue", label = 'Training')
autolabel(rectsP3,P3,TRperms[:,1])
rectsP3 = plt.bar(trials+width/2, TEperms[:,1], width, color="orange", label = 'Test')
autolabel(rectsP3,P3,TEperms[:,1])
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.title('Sensitivity Per Trial')
plt.ylabel('Sensitivity')
plt.xlabel('Trial') 
P4 = plt.subplot(224)
rectsP4 = plt.bar(trials-width/2, TRperms[:,2], width, color="blue", label = 'Training')
autolabel(rectsP4,P4,TRperms[:,2])
rectsP4 = plt.bar(trials+width/2, TEperms[:,2], width, color="orange", label = 'Test')
autolabel(rectsP4,P4,TEperms[:,2])
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.ylim([0,1.05])
plt.title('Specificity Per Trial')
plt.ylabel('Specificity')
plt.xlabel('Trial')  
plt.show()