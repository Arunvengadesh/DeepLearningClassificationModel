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
    
    step = 0.00005
     
    DATA_DIR = './data/fashion'
    
    mnist = LoadData.LoadDta(DATA_DIR,one_hot=True, reshape=False)
    
    X = tf.placeholder(tf.float32,[None,28,28,1])
    Y_ = tf.placeholder(tf.float32,[None,10])

    sess = tf.Session()
     
    #Random Number
    Randnum = random.uniform(0,0.5)
    #Randnum = 0.04526982



     

    #weights

    #W = tf.Variable(tf.zeros([784,10],tf.float32))
    W = tf.Variable(tf.truncated_normal([784,10],stddev = Randnum))

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
    lr = 0.0005 +  tf.train.exponential_decay(0.03, step, 2000, 1/math.e)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

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
    print('Singe Layer Traing Started>...............................')
    print('First Iteretion Number **************')
    for i in range(1,NumOfIteration):
    
        
            
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(DataLength)

        
        #print(i)
        
        sess.run(train_step,feed_dict = {X: batch_X,Y_: batch_Y})
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: batch_X, Y_: batch_Y})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        #TestAttrGen,TestClasGen = GenerateData(DataLength)
        #c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})    
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B,accuracy], feed_dict={X: batch_X, Y_: batch_Y})    
        Testacc_Array.append(a)
        TestCrossEntropy.append(c)


    print('Singe Layer Traing completed')       

    #acc_Array.append(a)#
    print('c')


    print(c)
    print('y_Round')
    print(np.round(y,2))
    print('y_Ceil')
    #print(np.round(y) +' '+ y_)

    print(np.round(y))

    print('y_')
    print(y_)
    print('w')
    print(w)
    print('b')
    print(b)
    print('a')
    print(a)    
    return acc_Array,Testacc_Array,TrngCrossEntropy,TestCrossEntropy
           
            
    #attr_val,class_val = sess.run([X,Y],feed_dict={X: Attr, Y: clas})

    #print(attr_val)
    #print(class_val)

    #y = [[1.0],[3.605551275],[5.744562647],[7.810249676],[9.848857802],[11.87434209],[13.89244399],[15.90597372],[17.91647287],[19.92485885]]

