import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# simple computer vision dataset with images of handwritten numbers and their labels.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


'''
input > weight > hidden layer 1 (activation function) > weights
> hidden layer 2 (activcation function) > weights > output layer

compare output to intended output > cost or loss function (ie. cross entropy)
optimazation function (ie. optimizer) > minimize that cost (ie. AdamOptimizer, stochastic gradient descent, AdaGrad)

backpropagation: go backwards to manipulate weights.

feed forward + backprop = epoch 
'''




# 10 classes, 0-9
''' one_hot: only one element is on or "hot". 
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# x isn't a specific value, it's a placeholder.
# we want to input any number of MNIST images, even flattened to 784-dimensional vector.
# represent this as 2D tensor of floats, with a shape [None, 784]. None means that a dimension can be of any length.
x = tf.placeholder('float', [None, 784])  
y = tf.placeholder('float')


def neural_network_model(data):
    # biases allow some neurons to fire even if all inputs are 0.
    # create a tensor or array of your data using random numbers. 
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}
    


    # activation function
    # (input_data * weights) + biases

    # matrix multiplication
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
             # tf.nn.relu: an activation function that computes rectified linear and returns a tensor.                                          

                          # layer2's input is whatever the activation function returns for layer1
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output




def train_neural_network(x):
    prediction = neural_network_model(x)   # pass in data
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )   # cost var measures how wrong. should minimize by manipulating weights.    

    # optimize cost function by using AdamOptimizer. Other popular
    # optimizers are Stochastic gradient descent & AdaGrad.
    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    
    hm_epochs = 10  # how many epochs wanted (cycles of feed forward and back prop) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        '''
        for each epoch and batch in data, run optimizer & cost against batch of data.
        to keep track of loss/cost at each step, add total cost per epoch up. 
        '''
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)



        # how many predictions made that matched the labels.
        # compare prediction to actual label.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) 



train_neural_network(x)
'''
Epoch 0 completed out of 10 loss: 1452714.4154663086
Epoch 1 completed out of 10 loss: 373885.2536678314
Epoch 2 completed out of 10 loss: 201322.66721487045
Epoch 3 completed out of 10 loss: 115658.79535222054
Epoch 4 completed out of 10 loss: 71050.93447960287
Epoch 5 completed out of 10 loss: 47524.903540165426
Epoch 6 completed out of 10 loss: 30635.953351485812
Epoch 7 completed out of 10 loss: 22305.810112327337
Epoch 8 completed out of 10 loss: 18404.217231426766
Epoch 9 completed out of 10 loss: 14817.764628920704
Accuracy: 0.9469

95% accuracy is very bad compared to more popular methods that give 99%.
But considering the only information given to the network was pixel values,
    did not tell it how to look for patterns,
    did not tell it how to tell a 4 from a 9

The network figured it out with an inner model,
based purely on pixel values to start, and achieved 95% accuracy.   
'''
    
