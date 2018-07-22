import tensorflow as tf

a = tf.constant(1, tf.int32, name='a')
b = tf.Variable(2, tf.int32, name='b')
c = tf.placeholder(tf.int32, name='c')
sum1 = tf.add(a, b, name='sum1')
sum2 = tf.add(a, c, name='sum2')

assign = b.assign(3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(init)
    res1 = sess.run(sum1)
    print(res1)

    res2 = sess.run(sum2, feed_dict={c:3})
    print(res2)

writer.close()