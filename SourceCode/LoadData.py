import numpy as np
import tensorflow as tf
from trainer.task import load_dataset
from sklearn.utils import shuffle
from trainer.task import load_mnist_data
from tensorflow.examples.tutorials.mnist import input_data

class MnistData(object):

    def __init__(self, tf_dataset, one_hot, reshape):
        self.pos = 0
        self.images = None
        self.labels = None
        # load entire Dataset into memory by chunks of 10000
        tf_dataset = tf_dataset.batch(10000)
        tf_dataset = tf_dataset.repeat(1)
        features, labels = tf_dataset.make_one_shot_iterator().get_next()
        if not reshape:
            features = tf.reshape(features, [-1, 28, 28, 1])
        if one_hot:
            labels = tf.one_hot(labels, 10)
        with tf.Session() as sess:
            while True:
                try:
                    feats, labs = sess.run([features, labels])
                    self.images = feats if self.images is None else np.concatenate([self.images, feats])
                    self.labels = labs if self.labels is None else np.concatenate([self.labels, labs])
                except tf.errors.OutOfRangeError:
                    break


    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.images) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.images[self.pos:self.pos+batch_size], self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res

class Mnist(object):
    def __init__(self, train_dataset, test_dataset, one_hot, reshape):
        self.train = MnistData(train_dataset, one_hot, reshape)
        self.test = MnistData(test_dataset, one_hot, reshape)


def LoadDta(data_dir,one_hot, reshape):
     
    
    
    train_images_file, train_labels_file, test_images_file, test_labels_file = load_mnist_data(data_dir)
    
      
    train_dataset = load_dataset(train_images_file, train_labels_file)
    train_dataset = train_dataset.shuffle(60000)
    
    test_dataset = load_dataset(test_images_file, test_labels_file)
    
    
    mnist = Mnist(train_dataset, test_dataset, one_hot, reshape)

    return mnist 



