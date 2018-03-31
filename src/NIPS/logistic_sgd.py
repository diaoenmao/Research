"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # self.W = theano.shared(
            # value=numpy.zeros(
                # (n_in, n_out),
                # dtype=theano.config.floatX
            # ),
            # name='W',
            # borrow=True
        # )
        # initialize the biases b as a vector of n_out 0s
        # self.b = theano.shared(
            # value=numpy.zeros(
                # (n_out,),
                # dtype=theano.config.floatX
            # ),
            # name='b',
            # borrow=True
        # )

        self.param = theano.shared(
            value=np.zeros(
                ((n_in+1)*n_out,),
                dtype=theano.config.floatX
            ),
            name='param',
            borrow=True
        )
        self.input = input
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        
        matrix_param = self.param.reshape(((n_in+1, n_out)))
        self.W = matrix_param[:n_in,]
        #self.W.tag.test_value = np.random((784,10),dtype=theano.config.floatX)
        self.b = matrix_param[-1,]
        #self.b.tag.test_value = np.random((1,10),dtype=theano.config.floatX)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input

    def nll(self, y):
        return -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        
        
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def gradient(self,y):
        m_nll = self.negative_log_likelihood(y)
        return T.grad(m_nll, self.param)
        
    def updates(self,learning_rate,y):
        return [(self.param,self.param-learning_rate*self.gradient(y))]

    def grad_loop(self,t_nll,A):
        #c_int = T.cast(c, 'int32')
        #c_nll = t_nll[i]
        #print(t_nll.type)
        return A+T.grad(t_nll,self.param)
        
    def TIC_gradient(self,y):
        # theano.config.compute_test_value = 'off'
        #t_nll = self.nll(y)
        #tmp = T.dvector('tmp')
        #outputs_info= T.dmatrix(')
        # #print(x.shape.eval())
        # x = np.zeros(((785)*10,),dtype=theano.config.floatX)
        # tmp_y = T.ivector('tmp_y')
        # #tmp_y.tag.test_value = np.random.randint(10, size=(600,))
        # #print(self.nll(tmp_y))
        # # self.nll(np.ones((5,),dtype='int32'))
        # g_theta, updates_theta = theano.scan(lambda i,tmp_y,x : T.grad(self.nll(tmp_y)[0], x), outputs_info=None, sequences=T.arange(tmp_y.shape[1]), non_sequences=[tmp_y,x])
        # sum_g_theta = g_theta.sum()
        # G_theta = theano.function(inputs=[tmp_y,x], outputs=sum_g_theta, updates = updates_theta)/self.input.shape[0]        
        # return G_theta(y,self_param.get_value())
        # tmp_y = T.ivector('tmp_y')
        # tmp_nll = -T.log(self.p_y_given_x)[T.arange(tmp_y.shape[0]), tmp_y]
        # tmp_param = theano.shared(np.zeros((785*10,)).astype(dtype='float64'))
        
        # g_theta, updates_theta = theano.scan(lambda i: T.grad(tmp_nll[i],tmp_param),sequences=[T.arange(tmp_nll.shape[0])])
        # sum_g_theta = g_theta.sum()
        # G_theta = theano.function(inputs=[tmp_y,tmp_param], outputs=sum_g_theta, updates = updates_theta, allow_input_downcast=True)/self.input.shape[0]
        
        
        #print(y.type)
        # start = T.dmatrix('start')
        # t_nll = T.dvector('t_nll')
        # t_y = T.ivector('t_y')
        # t_nll = self.nll(t_y)
        # print(t_nll.shape.eval({t_y: int(np.ones((5,),dtype=np.int))}))
        # p=T.dvector('p')

        # results, _ = theano.scan(lambda result, t_nll, p: result + T.dot(T.grad(t_nll,p),T.grad(t_nll,p).T),
                                 # outputs_info=[start], sequences = [t_nll], non_sequences=[p])

        # f = function([t_y,p], results[-1])
        # f([y,self.param.get_value()])
        
        n = 1000
        t_nll = self.nll(y)
        #print(self.input.shape[0])
        tmp = T.zeros((self.input.shape[1]+1,self.input.shape[1]+1))
        for i in range(n):
            #print(i)
            tmp=tmp+T.dot(T.grad(t_nll[i],self.param),T.grad(t_nll[i],self.param).T)
        return tmp/self.input.shape[0]
        
        
        #t_nll = self.nll(y)
        # tmp_y = T.ivector('tmp_y')
        # tmp_nll=self.nll(tmp_y)
        # x=T.dvector('x')
        # g_theta, updates_theta = theano.scan(lambda tmp_nll,x: T.grad(tmp_nll,x),sequences=tmp_nll,non_sequences=[x])
        # G_theta = theano.function(inputs=[tmp_y,x], outputs=g_theta, updates = updates_theta, allow_input_downcast=True)/self.input.shape[0]
        # a = G_theta(y,self.param)
        
    def TIC_hessian(self,y):
        sum_nll = -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # print(sum_nll.shape.eval())
        # x=self.param
        # gy = T.grad(sum_nll, self.param)
        # print(gy.eval())
        # h_theta, updates_theta = theano.scan(lambda v_gy, x : T.grad(v_gy, x), sequences=gy, non_sequences=x)
        # H_theta = theano.function(gy, h_theta, updates=updates)/self.input.shape[0]
        H_theta = theano.gradient.hessian(sum_nll,self.param)/self.input.shape[0]
        return H_theta
        
    def TIC(self,y):
        inverse = T.nlinalg.matrix_inverse(self.TIC_hessian(y))
        print('kkkk')
        gr = self.TIC_gradient(y)
        print('aaa')
        dooo = T.dot(inverse,gr)
        print('bbb')
        out = T.nlinalg.trace(dooo)/self.input.shape[0]
        print('ccc')
        return out
        #return T.dot(matrix_inverse(self.TIC_hessian(y)),self.TIC_gradient(y))
        
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=1000):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    penalty = classifier.TIC(y)
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    #gg = classifier.gradient(y)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = classifier.updates(learning_rate,y)

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break
    #_,minibatch_avg_penalty = train_model(minibatch_index)
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
    # print(
        # (
            # 'Penalty is %f,'
        # )
        # % minibatch_avg_penalty
    # )      
    


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl','rb'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print("Actual values")
    # for i in range(test_set_y.shape[0].eval()):
        # print(test_set_y[i].eval())
    print(test_set_y[:10].eval())

def a():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
# We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[0]
    print(test_set_y.eval().dtype)
    # load the saved model
    classifier = pickle.load(open('best_model.pkl','rb'))
    x = T.matrix('x')
    x= classifier.input
    y =T.ivector('y')
    batch_size = 1000
    index = T.lscalar()
    # compile a predictor function
    print('123131')
    test_model = theano.function(
        inputs=[x,y],
        outputs=classifier.TIC(y),
    )
    
    #test_set_x = test_set_x.get_value()
    #print(test_set_y)
    print('pop')
    predicted_values = test_model(test_set_x.get_value()[:,1],test_set_y.eval())
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    #print("Actual values")
    # for i in range(test_set_y.shape[0].eval()):
        # print(test_set_y[i].eval())
    #print(test_set_y[500:1600].eval())
    
if __name__ == '__main__':
    sys.setrecursionlimit(3000)
    #sgd_optimization_mnist()
    #predict()
    a()