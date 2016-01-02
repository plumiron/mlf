import random
import numpy as np
import matplotlib.pyplot as plt


class PLA(object):

    def __init__(self):
        self.w = None

    def fit(self, X, Y):
        # initialize w
        self.w = np.array([0.] * len(X[0]))

        # find a weight vector 'w' close to the perfect one 'wf'
        it = 0  # number of iterations
        is_no_error = False
        while not is_no_error:
            is_no_error = True
            for x, y in zip(X, Y):  # traverse samples
                y_pred = int(np.sign(np.dot(self.w, x))) or random.choice([-1, 1])
                if y_pred != y:  # prediction error
                    self.w = self.w + y * x  # fix w
                    is_no_error = False  # need traverse one more time
                    break
            it += 1
            print 'After %d iterations, w = %s' % (it, str(self.w))
        print 'Best w = %s' % str(self.w)

    def predict(self, X):
        if self.w is None:  # not fit yet
            raise ValueError('You need fit data first.')
        return np.array([int(np.sign(np.dot(self.w, x))) or random.choice([-1, 1])
                         for x in X])


def generate_samples(n_samples, n_features, w=None):
    if w is None:
        w = np.random.uniform(-1., 1., n_features)  # random w

    X = []
    Y = []
    while n_samples > 0:
        x = np.random.uniform(-5., 5., n_features)
        s = np.dot(w, x)
        if s != 0.:
            X.append(x)
            Y.append(np.sign(s))
            n_samples -= 1

    return np.array(X), np.array(Y)


if __name__ == '__main__':

    # generate samples
    n_samples = 100
    n_features = 2

    wf = np.random.uniform(-1., 1., n_features)
    X, Y = generate_samples(n_samples, n_features, wf)

    # run PLA
    clf = PLA()
    clf.fit(X, Y)

    # plot
    w = clf.w
    X1_w = X[:, 0]
    X2_w = [- w[0] / w[1] * x1 for x1 in X1_w]
    X1_wf = X[:, 0]
    X2_wf = [- wf[0] / wf[1] * x1_wf for x1_wf in X1_wf]

    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, marker='o', linewidth=0., label='sample')
    line_w = plt.plot(X1_w, X2_w, color='r', label='w')
    line_wf = plt.plot(X1_wf, X2_wf, color='g', label='wf')

    plt.xlim(-6., 6.)
    plt.ylim(-6., 6.)
    plt.title('PLA')
    plt.legend()
    plt.show()
