import numpy as np


def sigmo(x, derv=False):
    if derv is True:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
# seed


np.random.seed(1)

# synapses

syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# training


for j in xrange(60000):
    # layers
    l0 = x
    l1 = sigmo(np.dot(l0, syn0))
    l2 = sigmo(np.dot(l1, syn1))

    # backpropagation

    l2_error = y - l2
    if (j % 10000) == 0:
        print 'Er' + str(np.mean(np.abs(l2_error)))
        print l2
    l2_delta = l2_error * sigmo(l2, derv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmo(l1, derv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print l2
