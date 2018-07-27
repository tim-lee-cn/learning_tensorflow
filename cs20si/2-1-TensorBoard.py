import tensorflow as tf

a = tf.constant(2, name = 'a')
b = tf.constant(3, name = 'b')
sum = tf.add(a, b, name = 'add')
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    print(sess.run(sum))

writer.close()
