File
---------------------------
1. ShapeRegcog.py
2. DataGen.py 
3. SingleLayer.py
4. TwoHiddenLayer.py 
5. TwoHiddenLayerRelu.py
6. TwoHiddenLayerDropOut.py

Function ProtoType
---------------------------
 
1. GenerateData(size) -> return Attr,classi
2. SingleLayer(NumOfIteration) -> Return SingleLayerTrainAcc,SingleLayerTestAcc
3. TwoHiddenLayer(NumOfIteration) -> Return TwoHiddenLayerTrainAcc,TwoHiddenLayerTestAcc
4. TwoHiddenLayerRelu(NumOfIteration) -> Return TwoHiddenLayerReluTrainAcc,TwoHiddenLayerReluTestAcc
5. TwoHiddenLayerDropOut(NumOfIteration) -> Return TwoHiddenLayerDropOutTrainAcc,TwoHiddenLayerDropOutTestAcc     


Pseudo Code for ShapeRegcog
___________________________

SingleLayerTrainAcc,SingleLayerTestAcc = SingleLayer(NumOfIteration)
TwoHiddenLayerTrainAcc,TwoHiddenLayerTestAcc = TwoHiddenLayer(NumOfIteration)
TwoHiddenLayerReluTrainAcc,TwoHiddenLayerReluTestAcc = TwoHiddenLayerRelu(NumOfIteration)
TwoHiddenLayerDropOutTrainAcc,TwoHiddenLayerDropOutTestAcc = TwoHiddenLayerDropOut(NumOfIteration)

Plot(SingleLayerAcc,TwoHiddenLayerAcc,TwoHiddenLayerReluAcc,TwoHiddenLayerDropOutAcc)

