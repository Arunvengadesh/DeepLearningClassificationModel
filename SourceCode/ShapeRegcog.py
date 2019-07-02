from NoHidenLayer import NoHidenLayerMtd
from TwoHidenLayer import TwoHidenLayerMtd

from FIveHidenLayerRelu import FiveHidenLayerReluMtd
 
from LoadData import LoadDta
import matplotlib.pyplot as plt
DataLength = 100

NumOfIteration = 2000

print('***************Load Data Started*********************')


SingleLayerTrainAcc,SingleLayerTestAcc,SingleLayerTrainCrossEntropy,SingleLayerTestCrossEntropy = NoHidenLayerMtd(DataLength, NumOfIteration)


#TwoLayerTrainAcc,TwoLayerTestAcc,TwoLayerTrainCrossEntropy,TwoLayerTestCrossEntropy = TwoHidenLayerMtd(DataLength, NumOfIteration)

#FiveLayerReluTrainAcc,FiveTwoLayerReluTestAcc,FiveLayerReluTrainCrossEntropy,FIveLayerReluTestCrossEntropy = FiveHidenLayerReluMtd(DataLength, NumOfIteration)
   

print('******************Data Loaded ************************')


print('****************************ShapeRegcog****************************')
print('---------------------SingleLayer Started---------------------------')


print('---------------------SingleLayer Completed---------------------------')


plt.figure(1)
plt.subplot(211)
#plt.plot(SingleLayerTrainAcc,'r',TwoLayerTrainAcc,'g',FiveLayerReluTrainAcc,'y')
plt.plot(SingleLayerTrainAcc,'r')
plt.ylabel('Accuracy')
#plt.xlabel('Iteration')
plt.title('Accuracy')
plt.grid(True)
#plt.show()
plt.subplot(212)
#plt.plot(SingleLayerTrainCrossEntropy,'r',TwoLayerTrainCrossEntropy,'g',FiveLayerReluTrainCrossEntropy,'y')
plt.plot(SingleLayerTrainCrossEntropy,'r')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.title('Cross Entropy')
plt.grid(True)
plt.show()