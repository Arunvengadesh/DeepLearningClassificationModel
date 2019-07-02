import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
print('Tensorflow Version ' + tf.__version__)
import random
from DataGen import GenerateData

tf.set_random_seed(0)
#placeholder
def TwoHidenLayerLogitcsMtd(DataLength,NumOfIteration):
    

    X = tf.placeholder(tf.float32,[DataLength,4])
    Y_ = tf.placeholder(tf.float32,[DataLength,2])
    step = tf.placeholder(tf.int32)
    step = 0.00005

    sess = tf.Session()
         
    # RandNum = random.uniform(0,5)
    RandNum = 4.332905754582126
    print('RandNum........')
    print(RandNum)
    #weights
    L1 = 8
    L2 = 2
    
    W1 = tf.Variable(tf.truncated_normal([4,L1], stddev=RandNum))
    #tf.Variable(tf.zeros([4,L1],tf.float32))

    #bias

    B1 = tf.Variable(tf.zeros([1,L1],tf.float32))        
    
    W2 = tf.Variable(tf.truncated_normal([L1,L2], stddev=RandNum))

    #bias

    B2 = tf.Variable(tf.zeros([1,L2],tf.float32))
    
    
    #Model
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    Ylogits = tf.matmul(Y1, W2) + B2
    Y = tf.nn.softmax(Ylogits)
    
    #Cross Entropy 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
    


    
    
    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # training, learning rate = 0.005
    lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

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
    print('Two Layer Traing Started>...............................')
    for i in range(1,NumOfIteration):
        
            
        AttrGen,ClasGen = GenerateData(DataLength)

        sess.run(train_step,feed_dict = {X: AttrGen,Y_: ClasGen})
        c, y,y_,w1,w2, b1,b2,a = sess.run([cross_entropy,Y,Y_, W1, W2, B1, B2,accuracy], feed_dict={X: AttrGen, Y_: ClasGen})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        TestAttrGen,TestClasGen = GenerateData(DataLength)
        c, y,y_,w1,w2, b1,b2,a = sess.run([cross_entropy,Y,Y_, W1, W2, B1,B2,accuracy], feed_dict={X: TestAttrGen, Y_: TestClasGen})    
        Testacc_Array.append(a)
        TestCrossEntropy.append(c)


    print('Two Layer Traing completed')    
    TestAttrGen,TestClasGen = GenerateData(DataLength)
    #a, c, w, b = sess.run([accuracy, cross_entropy, W, B], feed_dict={X: Attr, Y_: clas})    
    c, y,y_,w1,w2, b1,b2,a = sess.run([cross_entropy,Y,Y_, W1, W2, B1,B2,accuracy], feed_dict={X: TestAttrGen, Y_: TestClasGen})    

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
    print('w1')
    print(w1)
    print('w2')
    print(w2)
    print('b1')
    print(b1)
    print('b2')
    print(b2)
    print('a')
    print(a)    
    return acc_Array,Testacc_Array,TrngCrossEntropy,TestCrossEntropy
           
            
    #attr_val,class_val = sess.run([X,Y],feed_dict={X: Attr, Y: clas})

    #print(attr_val)
    #print(class_val)

    #y = [[1.0],[3.605551275],[5.744562647],[7.810249676],[9.848857802],[11.87434209],[13.89244399],[15.90597372],[17.91647287],[19.92485885]]

