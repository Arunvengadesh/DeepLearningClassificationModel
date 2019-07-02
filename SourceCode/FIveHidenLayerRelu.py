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
def FiveHidenLayerReluMtd(DataLength,NumOfIteration):
    
    
     
    DATA_DIR = './data/fashion'
    
    mnist = LoadData.LoadDta(DATA_DIR,one_hot=True, reshape=False)
    
    X = tf.placeholder(tf.float32,[None,28,28,1])
    Y_ = tf.placeholder(tf.float32,[None,10])

    sess = tf.Session()
    #RandNum = random.uniform(0,0.5)
    RandNum = 0.23197358164873916
    print('RandNum........')
    print(RandNum)     
    step = 0.00000005
    #weights
    L1 = 300
    L2 = 100
    L3 = 10
    #L4 = 15
    #L5 = 10

    W1 = tf.Variable(tf.truncated_normal([784,L1],stddev=RandNum))
    W2 = tf.Variable(tf.truncated_normal([L1,L2],stddev=RandNum))
    W3 = tf.Variable(tf.truncated_normal([L2,L3],stddev=RandNum))
    #W4 = tf.Variable(tf.truncated_normal([L3,L4],stddev=RandNum))
    #W5= tf.Variable(tf.truncated_normal([L4,L5],stddev=RandNum))

    #bias

    B1 = tf.Variable(tf.zeros([1,L1],tf.float32))
    B2 = tf.Variable(tf.zeros([1,L2],tf.float32))     
    B3 = tf.Variable(tf.zeros([1,L3],tf.float32))
    #B4 = tf.Variable(tf.zeros([1,L4],tf.float32))
    #B5 = tf.Variable(tf.zeros([1,L5],tf.float32))
    XX = tf.reshape(X, [-1, 784])
    #Model
    
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    #Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    #Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
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
    lr = 0.000001 +  tf.train.exponential_decay(0.03, step, 2000, 1/math.e)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

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
    print('Third Iteretion Number **************')
    for i in range(1,NumOfIteration):
    
        
            
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(DataLength)

        
        #print(i)
        
        sess.run(train_step,feed_dict = {X: batch_X,Y_: batch_Y})
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y})
        acc_Array.append(a)
        TrngCrossEntropy.append(c)
        #TestAttrGen,TestClasGen = GenerateData(DataLength)
        #c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W, B1,accuracy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})    
        c, y,y_,w, b,a = sess.run([cross_entropy,Y,Y_, W1, B1,accuracy], feed_dict={X: batch_X, Y_: batch_Y})    
        Testacc_Array.append(a)
        TestCrossEntropy.append(c)


    print(' Layer Traing completed')       

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
