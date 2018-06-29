import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import read_data
import tensorflow as tf
import time
import confusion_matrix

NUM_LABELS = 10
NUM_NEURONS = 50
LEARNING_RATE = 0.001
N_EPOCHS = 30

def main():
    filename = 'mnist_100.csv'
    train_perc = 0.7
    label_index = 0

    acc = []
    for i in range(100):
        print(i)
        (train_x, train_y), (test_x, test_y), possibleLabels = read_data.read_data(filename, train_perc, label_index, NUM_LABELS)

        numAttributes = len(train_x[0])
        numLabels = NUM_LABELS

        x = tf.placeholder(tf.float32, shape=[None, numAttributes])
        y = tf.placeholder(tf.float32, shape=[None, numLabels])

        W_hidden = tf.Variable(tf.truncated_normal([numAttributes, NUM_NEURONS], stddev=0.1))
        b_hidden = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))

        hidden_net = tf.matmul(x, W_hidden) + b_hidden
        hidden_out = tf.sigmoid(hidden_net)

        W_outlayer = tf.Variable(tf.truncated_normal([NUM_NEURONS, numLabels], stddev=0.1))
        b_outlayer = tf.Variable(tf.constant(0.1, shape=[numLabels]))

        output_net = tf.matmul(hidden_out, W_outlayer) + b_outlayer

        if numLabels == 1:
            predict = tf.sigmoid(output_net)
        else:
            predict = tf.nn.softmax(output_net)

        if numLabels == 1:
            cost = tf.reduce_sum(0.5 * (y - predict) * (y - predict))
        else:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_net))

        trainStep = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        with tf.Session() as sess:

            step = 0
            printEvery = 100
            maxIterations = 1000
            totalTime = 0

            sess.run(tf.global_variables_initializer())

            while step < maxIterations:
                step += 1

                # train the network
                startTime = time.process_time()
                sess.run(trainStep, feed_dict={x: train_x, y: train_y})
                totalTime += time.process_time() - startTime

                if step % printEvery == 0:
                    #p = sess.run(predict, feed_dict={x: train_x})

                    print("\nStep:", step, "\tTime:", totalTime / step)
                    #cm = confusion_matrix.buildConfusionMatrix(p, train_y, numLabels)
                    #confusion_matrix.printConfusionMatrix(cm, possibleLabels)
                    #print("Training:")
                    #confusion_matrix.printAccuracy(cm)

                    #print("Testing:")
                    #p = sess.run(predict, feed_dict={x: test_x})
                    #cm = confusion_matrix.buildConfusionMatrix(p, test_y, numLabels)
                    #confusion_matrix.printAccuracy(cm)

            p = sess.run(predict, feed_dict={x: test_x})

            #print("Confusion Matrix on Test Set:")
            cm = confusion_matrix.buildConfusionMatrix(p, test_y, numLabels)
            #confusion_matrix.printConfusionMatrix(cm, possibleLabels)

            #print("Average time:", totalTime / step)
            accuracy = confusion_matrix.printAccuracy(cm)
            acc.append(float(accuracy))

    print(sum(acc) / float(len(acc)))
    print (acc)

if __name__ == '__main__':
    main()
