import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# add layer
def add_layer(inputs, in_size, out_size, layer_name, activation_function):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) #initial bias is 0.1
   Wx_plus_b = tf.matmul(inputs, Weights) + biases  #matrix multiply
   Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
   outputs = activation_function(Wx_plus_b)
   tf.summary.histogram(layer_name+'/outputs',outputs)
   return outputs

def accuracy(prediction,labels):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accu

# loading data
dataset = np.loadtxt('single_robot_collision_avoidance.log')

#randomize the dataset
data_label = np.ndarray(shape=(600,), dtype=np.int32, buffer=np.array(range(0, 600), dtype=np.int32))
permutation = np.random.permutation(data_label.shape[0])
data = dataset[permutation, :]

#
data = data[:, 2:20]
trdata = data[0:400]
#vadata = data[400:500]
tsdata = data[400:600]

# training data
x_trdata = trdata[:, 2:len(trdata[0])]
y_trdata = trdata[:, 0:2]
# validation data
#x_vadata = vadata[:, 2:len(trdata[0])]
#y_vadata = vadata[:, 0:2]
# testing data
x_tsdata = tsdata[:, 2:len(trdata[0])]
y_tsdata = tsdata[:, 0:2]

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32,[None,16],name='x_input')  #type:float32
ys = tf.placeholder(tf.float32,[None,2],name='y_input')

# add hidden layer
l1 = add_layer(xs, 16, 16, 'l1', activation_function=tf.nn.sigmoid)
# add output layer
prediction = add_layer(l1, 16, 2, 'l2', activation_function=tf.nn.sigmoid)

# the loss between prediciton and real data
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_trdata - prediction),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

tr_accuracy = accuracy(prediction,y_trdata)
ts_accuracy = accuracy(prediction,y_tsdata)
#va_accuracy = accuracy(prediction,y_vadata)

sess = tf.Session()
merged = tf.summary.merge_all()

#summary writer
#train_writer = tf.summary.FileWriter("~/Desktop/",sess.graph)
#test_writer = tf.summary.FileWriter("logs/test",sess.graph)
sess.run(tf.global_variables_initializer())

#sess.run optimizer
for i in range(200):
   sess.run(train_step, feed_dict={xs: x_trdata, ys: y_trdata,keep_prob:0.5})
   if i % 50 == 0:
       #print(sess.run(loss, feed_dict={xs: x_trdata, ys: y_trdata,keep_prob:1}))
       #record loss
       #train_result = sess.run(merged, feed_dict={xs: x_trdata, ys: y_trdata,keep_prob:0.5})
       #test_result = sess.run(merged, feed_dict={xs: x_tsdata, ys: y_tsdata,keep_prob:0.5})
       #train_writer.add_summary(train_result,i)
       #test_writer.add_summary(train_result, i)
       train_accuracy = sess.run(tr_accuracy, feed_dict={xs: x_trdata, ys: y_trdata,keep_prob:1})
       print("step %d, training accuracy %g" % (i, train_accuracy))
       #validation_accuracy = sess.run(va_accuracy, feed_dict={xs: x_vadata, ys: y_vadata,keep_prob:1})
       #print("step %d, validation accuracy %g" % (i, validation_accuracy))

saver = tf.train.Saver()
saver.save(sess, "F:\Bristol\Robotics\Dissertation\code\model.ckpt", global_step=i)
print("Model restored.")

test_accuracy = sess.run(ts_accuracy, feed_dict={xs: x_tsdata, ys: y_tsdata, keep_prob: 1})
print("testing accuracy %g" % (test_accuracy))


