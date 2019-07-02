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
def TwoHidenLayerSigmodAndSoftmaxMomentumMtd(DataLength,NumOfIteration):
    
    
     
    DATA_DIR = './data/fashion'
    
    mnist = LoadData.LoadDta(DATA_DIR,one_hot=True, reshape=False)
    
    X = tf.placeholder(tf.float32,[None,28,28,1])
    Y_ = tf.placeholder(tf.float32,[None,10])

    sess = tf.Session()
    RandNum = random.uniform(0,0.7)
    #RandNum = 0.6124315340304713
    #RandNum = 0
    print('RandNum........')
    print(RandNum)     
    step = 0.00000005
    #step = 0.00001
    #weights
    L1 = 300
    L2 = 100
    L3 =10

    W1 = tf.Variable(tf.truncated_normal([784,L1],stddev=RandNum))
    W2 = tf.Variable(tf.truncated_normal([L1,L2],stddev=RandNum))
    W3 = tf.Variable(tf.truncated_normal([L2,L3],stddev=RandNum))

    #bias

    B1 = tf.Variable(tf.zeros([1,L1],tf.float32))
    B2 = tf.Variable(tf.zeros([1,L2],tf.float32))     
    B3 = tf.Variable(tf.zeros([1,L3],tf.float32))
    XX = tf.reshape(X, [-1, 784])
    #Model
    
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    #Y1 = tf.nn.softmax(tf.matmul(XX, W1) + B1)
    #Y2 = tf.nn.softmax(tf.matmul(Y1, W2) + B2)
    
    Ylogits = tf.matmul(Y2, W3) + B3
    Y = tf.nn.softmax(Ylogits)
    
    #Cross Entropy 
    #cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)


    
    
    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # training, learning rate = 0.005
    lr = 0.0001 +  tf.train.exponential_decay(0.03, step, 2000, 1/math.e)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    mom = 0.005
    
    train_step = tf.train.MomentumOptimizer(lr,mom).minimize(cross_entropy)

    #cost = tf.reduce_mean(tf.math.square(Y_ - Y))
    #cost = -tf.reduce_mean(Y_ * (Y))
    #train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) 
     
    TrngCrossEntropy = []
    TestCrossEntropy =[]
    Testacc_Array = []
    acc_Array = []
    print('Two HiddenLayer Sigmoid and softmax Momentum  Traing Started>...............................')
    print('Two Iteretion Number **************')
    for i in range(1,NumOfIteration):
    
        
            
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(DataLength)

        
        #print(i)
        
        sess.run(train_step,feed_dict = {X: batch_X,Y_: batch_Y})
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        #TestAttrGen,TestClasGen = GenerateData(DataLength)
        ct, y,y_,w, b,at = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: mnist.test.images[1:1000,:,:], Y_: mnist.test.labels[1:1000,:]})    
        #c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y})    
        Testacc_Array.append(at)
        TestCrossEntropy.append(ct)
        if ((i % 1000) == 0):
            print('Iteration ')
            print(i)
            print('Accuracy')
            print(at)
            print('Cross-Entropy')
            print(ct)
        

    print('Singe Layer Traing completed')       
   
    return acc_Array,Testacc_Array,TrngCrossEntropy,TestCrossEntropy
