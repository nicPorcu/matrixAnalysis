import sys
import numpy
import matplotlib.pyplot as plt
arguments=sys.argv
#paramsFile=open(arguments[1])
#paramsFile_2=open(arguments[2])
a1=numpy.load('paramsFile.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='latin1')
a2=numpy.load('paramsFile_2.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='latin1')
a1f16=numpy.copy(a1)
a2f16=numpy.copy(a2)

for x in range(95):
	a1f16[x]=numpy.array(a1[x], dtype=numpy.float16)
	a2f16[x]=numpy.array(a2[x], dtype=numpy.float16)

#a1=numpy.asarray(a1, dtype=numpy.float16)
#a2=numpy.asarray(a2, dtype=numpy.float16)


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
	if avDifferenceMatrix.max()!=bin(0) and index<3000:
		print(mse)
		plt.plot(index, mse, 'ro')
		meanSquareDifferenceRecursion(matrix1copy,matrix2copy, index+1, saveNum)
		pass
	else:
		string="Layer"+str(saveNum)+" 16BitMeanSquareDifferenceRecursion"
		plt.xlabel("Layer")
		plt.ylabel("Error")
		#plt.show()
		plt.savefig("MeanSquareDifferenceRec/"+string)
		plt.clf()

#for i in range(9,95):
	#meanSquareDifferenceRecursion(a1[i],a2[i],0,i)


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
		plt.xlabel("Layer")
		plt.ylabel("Error")
		rootMeanSquareDifferenceRecursion(matrix1copy,matrix2copy, index+1, saveNum)
		pass
	else:
		pass
		#plt.show()
		string="Layer"+str(saveNum)+" rootMeanSquareDifferenceRecursion"
		
		plt.savefig("RootMeanSquareDifferenceRec/"+string)
		plt.clf()
#for x in range(0,2):
	#rootMeanSquareDifferenceRecursion(a1[x],a2[x], 0, x)
#rootMeanSquareDifferenceRecursion(a1[0],a2[0],0)

def float16rootMeanSquareDifferenceRecursion(matrix1,matrix2, index, saveNum):
	matrix1copy=numpy.array(matrix1, dtype=numpy.float16)
	matrix2copy=numpy.array(matrix2, dtype=numpy.float16)
	sys.setrecursionlimit(index+100)
	avDifferenceMatrix=numpy.absolute(numpy.subtract(matrix1copy, matrix2copy))
	rmse = (numpy.mean((matrix1 - matrix2)**2))**0.5
	indexOfMax=numpy.argmax(avDifferenceMatrix)
	numpy.put(matrix1copy, indexOfMax, 0)
	numpy.put(matrix2copy, indexOfMax, 0)
	if numpy.count_nonzero(avDifferenceMatrix)!=0 and index<3000:
		print(index,rmse)
		plt.plot(index, rmse, 'ro')
		plt.title('float16')
		plt.xlabel("Layer")
		plt.ylabel("Error")
		float16rootMeanSquareDifferenceRecursion(matrix1copy,matrix2copy, index+1, saveNum)

		pass
	else:
		
		print("hi")
		string="Layer"+str(saveNum)+" 16bitRmsdRec"
		plt.savefig("16bitRootMeanSquareDifferenceRec/"+string)
		plt.clf()
#for x in range(0,95):
	#float16rootMeanSquareDifferenceRecursion(a1[x],a2[x], 0, x)




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
		#plt.show()
		pass

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
		
	plt.savefig("16BitMeanErrorAndVarianceByLayer")	
	#plt.show()
#plotFunctions()

def quantitizationMetric(matrix1, matrix2, numberOfSegments, layer):
	differenceMatrix=numpy.array(matrix1-matrix2, dtype=numpy.float16)
	avDifferenceMatrix=numpy.absolute(differenceMatrix, dtype=numpy.float16)
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
	#plt.savefig("16BitQuantatization/Layer"+str(layer)+" 16BitQuantatization")
	plt.clf()

	#print(arr)
	return arr

def quantitizationComparison(x):
	if(x%3==0):
		p=quantitizationMetric(a1[x],a2[x],16,x)
		q=quantitizationMetric(a1f16[x],a2f16[x],16,x)
	else:
		p=quantitizationMetric(a1[x],a2[x],4,x)
		q=quantitizationMetric(a1f16[x],a2f16[x],4,x)
	arr=numpy.absolute(p-q)
	print(arr)
	for i in range(len(arr)):
		plt.bar(i, arr[i])
	plt.title("Difference between 16 and 32 bit Quantitization Metric layer"+ str(x))
	plt.savefig("16and32bitQuantatizationComparison/Layer"+str(x)+" 16and32bitQuantatizationComparison")
	plt.clf()
	#plt.show()
for x in range(25):
	quantitizationComparison(x)