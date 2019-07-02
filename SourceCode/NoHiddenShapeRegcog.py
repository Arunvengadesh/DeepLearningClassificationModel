from NoHidenLayer import NoHidenLayerMtd
from TwoHidenLayer import TwoHidenLayerMtd
from matplotlib.ticker import NullFormatter
from FIveHidenLayerRelu import FiveHidenLayerReluMtd
import numpy as np
 
from LoadData import LoadDta
import matplotlib.pyplot as plt
DataLength = 100

NumOfIteration = 25000

print('***************Load Data Started*********************')


SingleLayerTrainAcc,SingleLayerTestAcc,SingleLayerTrainCrossEntropy,SingleLayerTestCrossEntropy = NoHidenLayerMtd(DataLength, NumOfIteration)


#TwoLayerTrainAcc,TwoLayerTestAcc,TwoLayerTrainCrossEntropy,TwoLayerTestCrossEntropy = TwoHidenLayerMtd(DataLength, NumOfIteration)

#FiveLayerReluTrainAcc,FiveTwoLayerReluTestAcc,FiveLayerReluTrainCrossEntropy,FIveLayerReluTestCrossEntropy = FiveHidenLayerReluMtd(DataLength, NumOfIteration)
   

print('******************Data Loaded ************************')


print('****************************ShapeRegcog****************************')
print('---------------------SingleLayer Started---------------------------')


print('---------------------SingleLayer Completed---------------------------')

x = np.arange(NumOfIteration-1)
plt.figure(1)
plt.subplot(211)
#plt.plot(SingleLayerTrainAcc,'r',TwoLayerTrainAcc,'g',FiveLayerReluTrainAcc,'y')
ax = plt.subplot(211)
ax.plot(x,SingleLayerTrainAcc,'r',label = 'No Hidden Layer')

ax.legend()
plt.ylabel('Accuracy')
#plt.xlabel('Iteration')
plt.title('Accuracy')
plt.grid(True)
#plt.show()
bx = plt.subplot(212)
#plt.plot(SingleLayerTrainCrossEntropy,'r',TwoLayerTrainCrossEntropy,'g',FiveLayerReluTrainCrossEntropy,'y')
bx.plot(x,SingleLayerTrainCrossEntropy,'r',label = 'No Hidden Layer')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.title('Cross Entropy')
plt.grid(True)
bx.legend()
plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()

