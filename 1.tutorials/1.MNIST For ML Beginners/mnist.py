
import tensorflow as tf
# download and install the data automatically
import input_data


# download dataset or open
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tensorflow placeholder
x = tf.placeholder("float", [None, 784])

# initialize both Wand bas tensors full of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# implement the model
# First, we multiply xby Wwith the expression tf.matmul(x,W). This is
# flipped from when we multiplied them in our equation, where we had , as
# a small trick to deal with xbeing a 2D tensor with multiple inputs. We
# then add b, and finally apply tf.nn.softmax.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy result place holder
y_ = tf.placeholder("float", [None, 10])


# cross entropy implementation
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# minimize cross_entropyusing the gradient descent algorithm with a
# learning rate of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables
init = tf.initialize_all_variables()

# launch the model in a Session, and run the operation that initializes
# the variables:
sess = tf.Session()
sess.run(init)

# run the training step 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# check if our prediction matches the truth.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))


# ask for our accuracy on our test data.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
