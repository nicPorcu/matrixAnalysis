import numpy,sys
import matplotlib.pyplot as plt
arguments=sys.argv
#paramsFile=open(arguments[1])
#paramsFile_2=open(arguments[2])
a1=numpy.load('paramsFile.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='latin1')
a2=numpy.load('paramsFile_2.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='latin1')


#Calculates the Mean Square difference of two matrices recursively, each time removing the point with the most error
#Each time it plots the point on a graph and at the end it saves them
"""
	:matrix1 the first matrix
	:matrix2 the second matrix
	:index, used to track the amount of recursion, initially 0
	:savenum used to save the figure
"""


def meanSquareDifferenceRecursion(matrix1,matrix2, index, saveNum):
	matrix1copy=numpy.copy(matrix1)
	matrix2copy=numpy.copy(matrix2)
	sys.setrecursionlimit(index+100)
	differenceMatrix=numpy.subtract(matrix1copy, matrix2copy)
	avDifferenceMatrix=numpy.absolute(differenceMatrix)
	mse = numpy.mean((matrix1 - matrix2)**2)
	indexOfMax=numpy.argmax(avDifferenceMatrix)
	numpy.put(matrix1copy, indexOfMax, 0)
	numpy.put(matrix2copy, indexOfMax, 0)
	if avDifferenceMatrix.max()!=0 and index<3000:
		print(mse)
		plt.plot(index, mse, 'ro')
		meanSquareDifferenceRecursion(matrix1copy,matrix2copy, index+1, saveNum)
		pass
	else:
		string="Layer"+str(saveNum)+" meanSquareDifferenceRecursion"
		
		plt.show()
		#plt.savefig(string)
		plt.clf()
meanSquareDifferenceRecursion(a1,a2,0,22)


#Calculates the Root Mean Square difference of two matrices recursively, each time removing the point with the most error
#Each time it plots the point on a graph and at the end it saves them

def rootMeanSquareDifferenceRecursion(matrix1,matrix2, index, saveNum):
	matrix1copy=numpy.copy(matrix1)
	matrix2copy=numpy.copy(matrix2)
	sys.setrecursionlimit(index+100)
	avDifferenceMatrix=numpy.absolute(numpy.subtract(matrix1copy, matrix2copy))
	rmse = (numpy.mean((matrix1 - matrix2)**2))**0.5
	indexOfMax=numpy.argmax(avDifferenceMatrix)
	numpy.put(matrix1copy, indexOfMax, 0)
	numpy.put(matrix2copy, indexOfMax, 0)
	if avDifferenceMatrix.max()!=0 and index<3000:
		print(index,rmse)
		plt.plot(index, rmse, 'ro')
		rootMeanSquareDifferenceRecursion(matrix1copy,matrix2copy, index+1)
		pass
	else:
		plt.show()
		#string="Layer"+str(saveNum)+" rootMeanSquareDifferenceRecursion"
		#plt.show()
		#plt.savefig(string)
		#plt.clf()
#for x in range(1):
	#rootmMeanSquareDifferenceRecursion(a1[x],a2[x], 0, x)
#rootMeanSquareDifferenceRecursion(a1[0],a2[0],0)




def meanPercentageErrorRecursion(matrix1,matrix2, index):
	matrix1copy=numpy.copy(matrix1)
	matrix2copy=numpy.copy(matrix2)
	sys.setrecursionlimit(index+100)
	differenceMatrix=numpy.subtract(matrix1copy, matrix2copy)
	avDifferenceMatrix=numpy.absolute(differenceMatrix)
	avmatrix1=numpy.absolute(matrix1)
	weightedErrorMatrix=avDifferenceMatrix/avmatrix1
	mse = numpy.mean((avDifferenceMatrix)/avmatrix1)
	indexOfMax=numpy.argmax(weightedErrorMatrix)
	numpy.put(matrix1copy, indexOfMax, 1)
	numpy.put(matrix2copy, indexOfMax, 1)
	if weightedErrorMatrix.max()!=0 and index<3000:
		print(mse)
		plt.plot(index, mse, 'ro')
		meanPercentageErrorRecursion(matrix1copy,matrix2copy, index+1)
		pass
	else:
		plt.show()

def maximumDistance(matrix1, matrix2):
	differenceMatrix=numpy.subtract(matrix1,matrix2)
	max=differenceMatrix.max()
	return max



def meanDifference(matrix1, matrix2):
	differenceMatrix=numpy.subtract(matrix1,matrix2)
	avDifferenceMatrix=numpy.absolute(differenceMatrix)
	return numpy.mean(avDifferenceMatrix)

def numDifferentWeights(matrix1, matrix2, threshold):
	differenceMatrix=numpy.subtract(matrix1, matrix2)
	avDifferenceMatrix=numpy.absolute(matrix1,matrix2)
	data=numpy.where(avDifferenceMatrix>threshold)
	return len(differenceMatrix[data])
	


	
def differenceVarienceFunction(matrix1, matrix2):

	differenceMatrix=numpy.subtract(matrix1, matrix2)
	avDifferenceMatrix=numpy.absolute(matrix1,matrix2)
	return numpy.var(avDifferenceMatrix)


def plotFunctions():
	for x in range(95):
		print(x, differenceVarienceFunction(a1[x],a2[x]), numDifferentWeights(a1[x],a2[x],1) )
		plt.plot(x,meanDifference(a1[x],a2[x]),'ro')
		a=differenceVarienceFunction(a1[x],a2[x])
		plt.plot(x, a, 'g^')
		
	plt.savefig("meanErrorAndVarianceByLayer")	
	plt.show()
#plotFunctions()

def quantitizationMetric(matrix1, matrix2, numberOfSegments, layer):
	differenceMatrix=matrix1-matrix2
	avDifferenceMatrix=numpy.absolute(differenceMatrix)
	max=avDifferenceMatrix.max()
	min=avDifferenceMatrix.min()
	dividerNumber=(max-min)/numberOfSegments
	arr=numpy.zeros(numberOfSegments, dtype=int)
	for x in numpy.nditer(avDifferenceMatrix):
		loc=int((x-min)/dividerNumber)
		if x!=max:
			arr[loc]=arr[loc]+1
	arr[numberOfSegments-1]+=1
	for x in range(len(arr)):
		plt.bar(x, arr[x])
	print(arr)
	plt.show()

for x in range(0):
	if(x%3==0):
		quantitizationMetric(a1[x],a2[x],16,x)
	else:
		quantitizationMetric(a1[x],a2[x],4,x)


