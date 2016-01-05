import random
import numpy as np
import matplotlib.pyplot as plt


class Pocket(object):

    def __init__(self, maxiter=100):
        self.maxiter = maxiter
        self.w = None

    def fit(self, X, Y):
        # initialize w
        self.w = np.array([0.] * len(X[0]))

        # find a weight vector 'w' close to the perfect one 'wf'
        it = 0  # number of iterations
        while it < self.maxiter:
            mistakes = []
            for x, y in zip(X, Y):  # traverse samples
                y_pred = int(np.sign(np.dot(self.w, x))) or random.choice([-1, 1])
                if y_pred != y:  # prediction error
                    mistakes.append((x, y))
            it += 1

            if len(mistakes) == 0:  # there is no mistake
                print 'After %d iterations, w = %s' % (it, str(self.w))
                break

            x, y = random.choice(mistakes)  # get a random mistake
            w = self.w + y * x  # try to fix w
            n_mistakes = 0
            for x, y in zip(X, Y):  # traverse samples
                y_pred = int(np.sign(np.dot(w, x))) or random.choice([-1, 1])
                if y_pred != y:  # prediction error
                    n_mistakes += 1
            if n_mistakes < len(mistakes):
                self.w = w  # do fix w
            print 'After %d iterations, w = %s' % (it, str(self.w))
        print 'Best w = %s' % str(self.w)

    def predict(self, X):
        if self.w is None:  # not fit yet
            raise ValueError('You need fit data first.')

        Y_pred = [int(np.sign(np.dot(self.w, x))) or random.choice([-1, 1]) for x in X]

        return np.array(Y_pred)


def generate_samples(n_samples, n_features, w=None):
    if w is None:
        w = np.random.uniform(-1., 1., n_features)  # random w

    X = [np.random.uniform(-5., 5., n_features) for _ in range(n_samples)]
    Y = [int(np.sign(np.dot(w, x))) or random.choice([-1, 1]) for x in X]

    # make dataset non-separaple
    n_flip = int(n_samples * 0.1)
    indexes = random.sample(range(n_samples), n_flip)
    for index in indexes:
        Y[index] = -Y[index]

    return np.array(X), np.array(Y)


if __name__ == '__main__':

    # generate samples
    n_samples = 100
    n_features = 2

    wf = np.random.uniform(-1., 1., n_features)
    X, Y = generate_samples(n_samples, n_features, wf)

    # run Pocket
    clf = Pocket(maxiter=100)
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
    plt.title('Pocket')
    plt.legend()
    plt.show()
