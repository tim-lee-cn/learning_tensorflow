import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_input = np.linspace(-1, 1, 100)
print(X_input.shape)
Y_input = X_input * 3 + np.random.randn(X_input.shape[0]) * 0.3

X = tf.placeholder(tf.float32, X_input.shape, name='X')
Y = tf.placeholder(tf.float32, X_input.shape, name='Y')
W = tf.Variable(0.0, dtype=tf.float32, name='W')
b = tf.Variable(0.0, dtype=tf.float32, name='b')

Y_predict = W * X + b

loss = np.square(Y - Y_predict)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(tf.global_variables_initializer())
    epochs = 100
    for _ in range(epochs):
        sess.run(optimizer, feed_dict={X: X_input, Y:Y_input})

    print(type(W))
    W, b = sess.run([W, b])
    print(type(W))

writer.close()

plt.plot(X_input, W * X_input + np.ones(X_input.shape) *  b, color='red')
plt.scatter(X_input, Y_input, color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('linear regression')
plt.show()

