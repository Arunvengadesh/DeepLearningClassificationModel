from NoHidenLayer import NoHidenLayerMtd
from TwoHidenLayerAllSoftmax import TwoHidenLayerMtd
from TwoHidenLayerSigmoidAndSoftmax import TwoHidenLayerSigmodAndSoftmaxMtd
from TwoHidenLayerSigmoidAndSoftmaxRMSProp import TwoHidenLayerSigmodAndSoftmaxRMSPropMtd

from TwoHidenLayerSigmoidAndSoftmaxAdaGrad import TwoHidenLayerSigmodAndSoftmaxAdaGradMtd
from TwoHidenLayerAdamOptSigmoidAndSoftmax import TwoHidenLayerAdamOptSigmoidAndSoftmaxMtd
from TwoHidenLayerSigmoidAndSoftmaxMomentum import TwoHidenLayerSigmodAndSoftmaxMomentumMtd

from TwoHidenLayerReluAndSoftmax import TwoHidenLayerReluAndSoftmaxMtd
from TwoHidenLayerAdamOptReluAndSoftmax import TwoHidenLayerAdamOptReluAndSoftmaxMtd
#from TwoHidenLayerAdamDecayOptSigmoidAndSoftmax import TwoHidenLayerAdamDecayOptSigmoidAndSoftmaxMtd
from TwoHidenLayerAdamDecayOptReluAndSoftmax import TwoHidenLayerAdamDecayOptReluAndSoftmaxMtd
import os
from TwoHidenLayerAdamDecayOptReluDropOutAndSoftmax import TwoHidenLayerAdamDecayOptReluDropOutAndSoftmaxMtd

from TwoHidenLayerRELUAndSoftmaxConvolution import TwoHidenLayerRELUAndSoftmaxConvolutionMtd


from matplotlib.ticker import NullFormatter

from FIveHidenLayerRelu import FiveHidenLayerReluMtd



import numpy as np
 
from LoadData import LoadDta
import matplotlib.pyplot as plt
DataLength = 100

NumOfIteration = 10000

print('***************Load Data Started*********************')

#********************************NoHidenLayerMtd*****************************************************
SingleLayerTrainAcc,SingleLayerTestAcc,SingleLayerTrainCrossEntropy,SingleLayerTestCrossEntropy = NoHidenLayerMtd(DataLength, NumOfIteration)

# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTrainAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTestAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTrainCrossEntropy.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTestCrossEntropy.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTrainAcc.txt', SingleLayerTrainAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTestAcc.txt', SingleLayerTestAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTrainCrossEntropy.txt', SingleLayerTrainCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\SingleLayerTestCrossEntropy.txt', SingleLayerTestCrossEntropy)


#*******************************TwoHidenLayerMtd All Softmax ******************************************************
#TwoLayerTrainAcc,TwoLayerTestAcc,TwoLayerTrainCrossEntropy,TwoLayerTestCrossEntropy = TwoHidenLayerMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainCrossEntropy.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestCrossEntropy.txt")

#np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAcc.txt', TwoLayerTrainAcc)
#np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAcc.txt', TwoLayerTestAcc)
#np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainCrossEntropy.txt', TwoLayerTrainCrossEntropy)
#np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestCrossEntropy.txt', TwoLayerTestCrossEntropy)

#********************************TwoHidenLayerSigmodAndSoftmaxMtd*******************************************************
#TwoLayerTrainSigmoidAcc,TwoLayerTestSigmoidAcc,TwoLayerTrainSigmoidCrossEntropy,TwoLayerTestSigmoidCrossEntropy = TwoHidenLayerSigmodAndSoftmaxMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropy.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropy.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAcc.txt', TwoLayerTestSigmoidAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAcc.txt', TwoLayerTrainSigmoidAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropy.txt', TwoLayerTrainSigmoidCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropy.txt', TwoLayerTestSigmoidCrossEntropy)

#*****************************TwoHidenLayerReluAndSoftmaxMtd***********************************************************
#TwoLayerTrainReluAcc,TwoLayerTestReluAcc,TwoLayerTrainReluCrossEntropy,TwoLayerTestReluCrossEntropy,r1,r2 = TwoHidenLayerReluAndSoftmaxMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainReluAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestReluAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainReluCrossEntropy.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestReluCrossEntropy.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainReluAcc.txt', TwoLayerTrainReluAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestReluAcc.txt', TwoLayerTestReluAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestReluCrossEntropy.txt', TwoLayerTestReluCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainReluCrossEntropy.txt', TwoLayerTrainReluCrossEntropy)

#******************************TwoHidenLayerAdamOptReluAndSoftmaxMtd***********************************************************
#TwoLayerTrainAdamRELUAcc,TwoLayerTestAdamRELUAcc,TwoLayerTrainAdamRELUCrossEntropy,TwoLayerTestAdamRELUCrossEntropy,Relu1,Relu2 = TwoHidenLayerAdamOptReluAndSoftmaxMtd(DataLength, NumOfIteration)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamRELUAcc.txt', TwoLayerTrainAdamRELUAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamRELUAcc.txt', TwoLayerTestAdamRELUAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamRELUCrossEntropy.txt', TwoLayerTrainAdamRELUCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamRELUCrossEntropy.txt', TwoLayerTestAdamRELUCrossEntropy)

#******************************************************************************************
#TwoLayerTrainAdamDecayReluAcc,TwoLayerTestAdamDecayReluAcc,TwoLayerTrainAdamDecayReluCrossEntropy,TwoLayerTestAdamDecayReluCrossEntropy= TwoHidenLayerAdamDecayOptReluAndSoftmaxMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamDecayReluAcc.txt")



# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluAcc.txt', TwoLayerTrainAdamDecayReluAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamDecayReluAcc.txt', TwoLayerTestAdamDecayReluAcc)

# TwoLayerTrainAdamDecayReluAcc = np.loadtxt("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluAcc.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluCrossEntropy.txt', TwoLayerTrainAdamDecayReluCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamDecayReluCrossEntropy.txt', TwoLayerTestAdamDecayReluCrossEntropy)
# TwoLayerTrainAdamDecayReluCrossEntropy = np.loadtxt("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluCrossEntropy.txt")


#********************************************************************************
#TwoLayerTrainAdamDecayReluDropOutAcc,TwoLayerTestAdamDecayReluDropOutAcc,TwoLayerTrainAdamDecayReluDropOutCrossEntropy,TwoLayerTestAdamDecayReluDropOutCrossEntropy = TwoHidenLayerAdamDecayOptReluDropOutAndSoftmaxMtd(DataLength, NumOfIteration)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluDropOutAcc.txt', TwoLayerTrainAdamDecayReluDropOutAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamDecayReluDropOutAcc.txt', TwoLayerTestAdamDecayReluDropOutAcc)
# #TwoLayerTrainAdamDecayReluDropOutAcc = np.loadtxt("E:\Study_materials\Semester-4\Project\SourceCode\Mid_Sem_Code\FashionMaster.01_backup\FashionMaster.01_backup\PlotData\TwoLayerTrainAdamDecayReluDropOutAcc.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamDecayReluDropOutCrossEntropy.txt', TwoLayerTrainAdamDecayReluDropOutCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamDecayReluDropOutCrossEntropy.txt', TwoLayerTestAdamDecayReluDropOutCrossEntropy)
#TwoLayerTrainAdamDecayReluDropOutCrossEntropy = np.loadtxt("E:\Study_materials\Semester-4\Project\SourceCode\Mid_Sem_Code\FashionMaster.01_backup\FashionMaster.01_backup\PlotData\TwoLayerTrainAdamDecayReluDropOutCrossEntropy.txt")



# #********************************TwoHidenLayerSigmodAndSoftmaxRMSPropMtd*******************************************************
# TwoLayerTrainSigmoidAccRMSProp,TwoLayerTestSigmoidAccRMSProp,TwoLayerTrainSigmoidCrossEntropyRMSProp,TwoLayerTestSigmoidCrossEntropyRMSProp = TwoHidenLayerSigmodAndSoftmaxRMSPropMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccRMSProp.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccRMSProp.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyRMSProp.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyRMSProp.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccRMSProp.txt', TwoLayerTrainSigmoidAccRMSProp)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccRMSProp.txt', TwoLayerTestSigmoidAccRMSProp)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyRMSProp.txt', TwoLayerTrainSigmoidCrossEntropyRMSProp)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyRMSProp.txt', TwoLayerTestSigmoidCrossEntropyRMSProp)
# #********************************TwoHidenLayerAdamOptSigmodAndSoftmax*******************************************************
#TwoLayerTrainAdamOptSigmoidAcc,TwoLayerTestAdamOptSigmoidAcc,TwoLayerTrainAdamOptSigmoidCrossEntropy,TwoLayerTestAdamOptSigmoidCrossEntropy = TwoHidenLayerAdamOptSigmoidAndSoftmaxMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamOptSigmoidAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamOptSigmoidAcc.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamOptSigmoidCrossEntropy.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamOptSigmoidCrossEntropy.txt")
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamOptSigmoidAcc.txt', TwoLayerTrainAdamOptSigmoidAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamOptSigmoidAcc.txt', TwoLayerTestAdamOptSigmoidAcc)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainAdamOptSigmoidCrossEntropy.txt', TwoLayerTrainAdamOptSigmoidCrossEntropy)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestAdamOptSigmoidCrossEntropy.txt', TwoLayerTestAdamOptSigmoidCrossEntropy)
# #******************************AdaGrad Opt*********************************
#TwoLayerTrainSigmoidAccAdaGrad,TwoLayerTestSigmoidAccAdaGrad,TwoLayerTrainSigmoidCrossEntropyAdaGrad,TwoLayerTestSigmoidCrossEntropyAdaGrad = TwoHidenLayerSigmodAndSoftmaxAdaGradMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyAdaGrad.txt")
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccAdaGrad.txt', TwoLayerTrainSigmoidAccAdaGrad)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccAdaGrad.txt', TwoLayerTestSigmoidAccAdaGrad)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyAdaGrad.txt', TwoLayerTrainSigmoidCrossEntropyAdaGrad)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyAdaGrad.txt', TwoLayerTestSigmoidCrossEntropyAdaGrad)

# #******************************Momentum Opt*********************************
#TwoLayerTrainSigmoidAccMomentum,TwoLayerTestSigmoidAccMomentum,TwoLayerTrainSigmoidCrossEntropyMomentum,TwoLayerTestSigmoidCrossEntropyMomentum = TwoHidenLayerSigmodAndSoftmaxMomentumMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyAdaGrad.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyAdaGrad.txt")
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidAccMomentum.txt', TwoLayerTrainSigmoidAccMomentum)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidAccMomentum.txt', TwoLayerTestSigmoidAccMomentum)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainSigmoidCrossEntropyMomentum.txt', TwoLayerTrainSigmoidCrossEntropyMomentum)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestSigmoidCrossEntropyMomentum.txt', TwoLayerTestSigmoidCrossEntropyMomentum)
#********************************Convolutional*********************************
#TwoLayerTrainRELUAccConvolution,TwoLayerTestRELUAccConvolution,TwoLayerTrainRELUCrossEntropyConvolution,TwoLayerTestRELUCrossEntropyConvolution = TwoHidenLayerRELUAndSoftmaxConvolutionMtd(DataLength, NumOfIteration)
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainRELUAccConvolution.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestRELUAccConvolution.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainRELUCrossEntropyConvolution.txt")
# os.remove("E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestRELUCrossEntropyConvolution.txt")

# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainRELUAccConvolution.txt', TwoLayerTrainRELUAccConvolution)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestRELUAccConvolution.txt', TwoLayerTestRELUAccConvolution)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTrainRELUCrossEntropyConvolution.txt', TwoLayerTrainRELUCrossEntropyConvolution)
# np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\TwoLayerTestRELUCrossEntropyConvolution.txt', TwoLayerTestRELUCrossEntropyConvolution)




print('******************Data Loaded ************************')
#while (TwoLayerTrainSigmoidAcc[NumOfIteration-2] < 0.9):
#    TwoLayerTrainSigmoidAcc,TwoLayerTestSigmoidAcc,TwoLayerTrainSigmoidCrossEntropy,TwoLayerTestSigmoidCrossEntropy = TwoHidenLayerSigmodAndSoftmaxMtd(DataLength, NumOfIteration)


print('****************************ShapeRegcog****************************')
print('---------------------SingleLayer Started---------------------------')

#SingleLayerTrainAcc,SingleLayerTestAcc,SingleLayerTrainCrossEntropy,SingleLayerTestCrossEntropy
#TwoLayerTrainAcc,TwoLayerTestAcc,TwoLayerTrainCrossEntropy,TwoLayerTestCrossEntropy
#TwoLayerTrainSigmoidAcc,TwoLayerTestSigmoidAcc,TwoLayerTrainSigmoidCrossEntropy,TwoLayerTestSigmoidCrossEntropy
#TwoLayerTrainReluAcc,TwoLayerTestReluAcc,TwoLayerTrainReluCrossEntropy,TwoLayerTestReluCrossEntropy
#TwoLayerTrainAdamRELUAcc,TwoLayerTestAdamRELUAcc,TwoLayerTrainAdamRELUCrossEntropy,TwoLayerTestAdamRELUCrossEntropy
#TwoLayerTrainAdamDecayReluAcc,TwoLayerTestAdamDecayReluAcc,TwoLayerTrainAdamDecayReluCrossEntropy,TwoLayerTestAdamDecayReluCrossEntropy
#TwoLayerTrainSigmoidAccAdaGrad,TwoLayerTestSigmoidAccAdaGrad,TwoLayerTrainSigmoidCrossEntropyAdaGrad,TwoLayerTestSigmoidCrossEntropyAdaGrad
#TwoLayerTrainAdamOptSigmoidAcc,TwoLayerTestAdamOptSigmoidAcc,TwoLayerTrainAdamOptSigmoidCrossEntropy,TwoLayerTestAdamOptSigmoidCrossEntropy
#TwoLayerTrainSigmoidAccMomentum,TwoLayerTestSigmoidAccMomentum,TwoLayerTrainSigmoidCrossEntropyMomentum,TwoLayerTestSigmoidCrossEntropyMomentum
#TwoLayerTrainRELUAccConvolution,TwoLayerTestRELUAccConvolution,TwoLayerTrainRELUCrossEntropyConvolution,TwoLayerTestRELUCrossEntropyConvolution
print('---------------------SingleLayer Completed---------------------------')

# x = np.arange(NumOfIteration-1)
# plt.figure(1)
# plt.plot(x,SingleLayerTestAcc,label = 'No Hidden Layer Test ACC')
# #plt.plot(x,TwoLayerTestAcc,label = 'Two Hidden Layer All Softmax Test ACC')
# #plt.plot(x,TwoLayerTestSigmoidAcc,label = 'Two Hidden Layer Sigmoid Test ACC')
# #plt.plot(x,TwoLayerTestReluAcc,label = 'Two Hidden Layer RELU Test ACC')
# #plt.plot(x,TwoLayerTestAdamRELUAcc,label = 'Two Hidden Layer Adam RELU Test ACC')
# #plt.plot(x,TwoLayerTestAdamDecayReluAcc,label = 'Two Hidden Layer Adam Decay RELU Test ACC')
# #plt.plot(x,TwoLayerTestSigmoidAccAdaGrad,label = 'Two Hidden Layer Adagrad Test ACC')
# #plt.plot(x,TwoLayerTestAdamOptSigmoidAcc,label = 'Two Hidden Layer AdamOpt Test ACC')
# #plt.plot(x,TwoLayerTestSigmoidAccMomentum,label = 'Two Hidden Layer Momentum Test ACC')
# #plt.plot(x,TwoLayerTestRELUAccConvolution,label = 'Two Hidden Layer Convolution Test ACC')
# plt.legend()
# plt.ylabel('Accuracy')
# #plt.ylim(0,0.1,1)

# #plt.xlabel('Iteration')
# plt.title('Accuracy')
# plt.grid(True)
# #plt.show()
# plt.figure(2)
# plt.plot(x,SingleLayerTestCrossEntropy,label = 'No Hidden Layer Test Cross Entropy')
# #plt.plot(x,TwoLayerTestCrossEntropy,label = 'Two Hidden Layer All Softmax Test Cross Entropy')
# #plt.plot(x,TwoLayerTestSigmoidCrossEntropy,label = 'Two Hidden Layer Sigmoid Test Cross Entropy')
# #plt.plot(x,TwoLayerTestReluCrossEntropy,label = 'Two Hidden Layer RELU Test Cross Entropy')
# #plt.plot(x,TwoLayerTestAdamRELUCrossEntropy,label = 'Two Hidden Layer Adam RELU Test Cross Entropy')
# #plt.plot(x,TwoLayerTestAdamDecayReluCrossEntropy,label = 'Two Hidden Layer Adam Decay RELU Test Cross Entropy')
# #plt.plot(x,TwoLayerTestSigmoidCrossEntropyAdaGrad,label = 'Two Hidden Layer AdaGrad Test Cross Entropy')
# #plt.plot(x,TwoLayerTestAdamOptSigmoidCrossEntropy,label = 'Two Hidden Layer AdamOpt Test Cross Entropy')
# #plt.plot(x,TwoLayerTestSigmoidCrossEntropyMomentum,label = 'Two Hidden Layer Momentum Test Cross Entropy')
# #plt.plot(x,TwoLayerTestRELUCrossEntropyConvolution,label = 'Two Hidden Layer Convolution Test Cross Entropy')

# plt.ylabel('Cross Entropy')
# plt.xlabel('Iteration')
# #plt.ylim(0,1)
# plt.title('Cross Entropy')
# plt.grid(True)
# plt.legend()
# plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
# plt.show()

