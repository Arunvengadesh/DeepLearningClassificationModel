import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
print('Tensorflow Version ' + tf.__version__)
import random
from DataGen import GenerateData
import LoadData

tf.set_random_seed(0)
#placeholder
def NoHidenLayerMtd(DataLength,NumOfIteration):
    
    
   
    
    DATA_DIR = './data/fashion'
    
    mnist = LoadData.LoadDta(DATA_DIR,one_hot=True, reshape=False)
    
    X = tf.placeholder(tf.float32,[None,28,28,1])
    Y_ = tf.placeholder(tf.float32,[None,10])

    sess = tf.Session()
         

    #weights

    W = tf.Variable(tf.zeros([784,10],tf.float32))

    #bias

    B = tf.Variable(tf.zeros([1,10],tf.float32)) 
    XX = tf.reshape(X, [-1, 784])
    #Model
    Y = tf.nn.softmax(tf.matmul(XX, W) + B)        
    #Cross Entropy 
    cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))


    
    
    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

    #cost = tf.reduce_mean(tf.math.square(Y_ - Y))
    #cost = -tf.reduce_mean(Y_ * (Y))
    #train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cross_entropy)
    
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("Weights", W)
    tf.summary.histogram("Bios", B)
    
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("E:\\Study_materials\\Semester-4\\Project\\SourceCode\\Mid_Sem_Code\\FashionMaster25022019_backup\\FashionMaster02_backup\\tmp\\mnist_demo\\3")
    writer.add_graph(sess.graph)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) 
     
    TrngCrossEntropy = []
    TestCrossEntropy =[]
    Testacc_Array = []
    acc_Array = []
    
    print('No Hidden Layer Traing Started>...............................')
    print('First Iteretion Number **************')
    for i in range(1,NumOfIteration):
    
        
            
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(DataLength)
        batchShapeX = batch_X.shape
        batchShapeY = batch_Y.shape



        _, s = sess.run([train_step,merged_summary],feed_dict = {X: batch_X,Y_: batch_Y})
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: batch_X, Y_: batch_Y})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        #TestAttrGen,TestClasGen = GenerateData(DataLength)
        ct, y,y_,w, b,at = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: mnist.test.images[1:1000,:,:], Y_: mnist.test.labels[1:1000,:]})    
        #c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: batch_X, Y_: batch_Y})    
        Testacc_Array.append(at)
        TestCrossEntropy.append(ct)
        
        
        
#        s = sess.run(merged_summary,feed_dict = {X: batch_X,Y_: batch_Y})
        writer.add_summary(s, i)
        
        # if ((i % 1000) == 0):
            # print('Iteration ')
            # print(i)
            # print('Accuracy')
            # print(at)
            # print('Cross-Entropy')
            # print(ct)

    print('No Hidden Layer Traing completed')
    np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\W1.txt', w)
    np.savetxt('E:\\Study_materials\\Semester-4\\Project\\FinalSemProgress\\PlotData\\B1.txt', b)
    
    
    return acc_Array,Testacc_Array,TrngCrossEntropy,TestCrossEntropy
           
            
    