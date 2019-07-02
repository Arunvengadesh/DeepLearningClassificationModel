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
def TwoHidenLayerRELUAndSoftmaxConvolutionMtd(DataLength,NumOfIteration):
    
    
     
    DATA_DIR = './data/fashion'
    
    mnist = LoadData.LoadDta(DATA_DIR,one_hot=True, reshape=False)
    
    X = tf.placeholder(tf.float32,[None,28,28,1])
    Y_ = tf.placeholder(tf.float32,[None,10])

    sess = tf.Session()
    #RandNum = random.uniform(0,0.7)
    #RandNum = 0.6124315340304713
    RandNum = 0.1
    print('RandNum........')
    print(RandNum)     
    #step = 0.00000005
    step = tf.placeholder(tf.int32)
    #weights
    L1 = 4
    L2 = 8
    L3 =12
    L4 = 200
    L5 = 10

    W1 = tf.Variable(tf.truncated_normal([5,5,1,L1],stddev=RandNum))
    W2 = tf.Variable(tf.truncated_normal([5,5,L1,L2],stddev=RandNum))
    W3 = tf.Variable(tf.truncated_normal([4,4,L2,L3],stddev=RandNum))
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * L3, L4], stddev=RandNum))
    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=RandNum))

    #bias

    B1 = tf.Variable(tf.ones([L1])/10)
    B2 = tf.Variable(tf.ones([L2])/10)     
    B3 = tf.Variable(tf.ones([L3])/10)
    B4 = tf.Variable(tf.ones([L4])/10)
    B5 = tf.Variable(tf.ones([L5])/10)
    #XX = tf.reshape(X, [-1, 784])
    #Model
    stride = 1
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
    
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4) 
    
    #Y1 = tf.nn.softmax(tf.matmul(XX, W1) + B1)
    #Y2 = tf.nn.softmax(tf.matmul(Y1, W2) + B2)
    
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)
    
    #Cross Entropy 
    #cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)


    
    
    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # training, learning rate = 0.005
    lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

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
    print('Two HiddenLayer RELU and softmax Convolution Traing Started>...............................')
    print('Two Iteretion Number **************')
    for i in range(1,NumOfIteration):
    
        
            
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(DataLength)

        print("I Value = ")
        print(i)
        
        
        sess.run(train_step,feed_dict = {X: batch_X,Y_: batch_Y, step: i})
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y, step: i})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        #TestAttrGen,TestClasGen = GenerateData(DataLength)
        ct, y,y_,w, b,at = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: mnist.test.images[1:1000,:,:], Y_: mnist.test.labels[1:1000,:], step: i})    
        #c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y})    
        Testacc_Array.append(at)
        TestCrossEntropy.append(ct)
        print("Testacc_Array")
        print(at)

    print('Two HiddenLayer RELU and softmax Convolution Traing completed')       

     
    return acc_Array,Testacc_Array,TrngCrossEntropy,TestCrossEntropy
           
            
    
