import random
import math
import operator
import numpy as np
import struct
import time

# It is recommended that one read the README before using this class

class KanervaCoder:
    distanceMeasure = 'euclidian'  # alternatively hamming distance could be used
    numPrototypes = 50
    dimensions = 1
    threshold = 0.02

    # an alternative to the threshold is to take the X closest points
    numClosest = 10
    prototypes = None
    visitCounts = None
    updatePrototypes = None
    minNumberVisited = 50

    Fuzzy = False

    updateFunc = None

    activationRadii = 0
    beenAroundTheBlock = False  # set to true once a single prototype has been visited the minNumberVisited

    def __init__(self, _numPrototypes, _dimensions, _distanceMeasure):
        if _distanceMeasure != 'hamming' and _distanceMeasure != 'euclidian':
            raise AssertionError('Unknown distance measure ' + str(_distanceMeasure) + '. Use hamming or euclidian.')
            return
        if _numPrototypes < 0:
            raise AssertionError('Need more than 2 prototypes. ' + str(
                _numPrototypes) + ' given. If 0 given, 50 prototpyes are used by default.')
            return

        self.dimensions = _dimensions
        self.distanceMeasure = _distanceMeasure
        self.numPrototypes = _numPrototypes

        # because each observation is normalized within its range,
        # each prototype can be a random vector where each dimension is within (0-1)
        self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])

        # this is a counter for each prototype that increases each time a prototype is visited
        self.visitCounts = np.zeros(self.numPrototypes)

        # this is used within the learner to manipulate the prototype location
        self.updatedPrototypes = []

        # this is one thing I have been testing, if we want to manipulate our prototypes (combine/move/add)
        # we should make sure that we have explored the state space sufficinetly enough
        # minNumberVisited is one way to specify that we want at least one prototype to be visited
        # this number of times before we manipulate our representation
        self.minNumberVisited = self.numPrototypes / 2

        # if updateFunc is 0, perform the representation update function found in the XGame paper
        # if updateFunc is 1, perform the representation update function found in the Case Studies paper
        self.updateFunc = 1

        # the activationRadii is defined in the Case Study paper and is used as a radius to find all
        # prototypes that are sufficiently close.
        # setting the activationRadii to 0 will include no prototypes, and to 1 will include all of them
        self.activationRadii = .1

        # This is defined in the Case Study paper as a way of limiting how many prototypes should are activated
        # by a given observation
        self.caseStudyN = 5

        # if false, an array of the indexes of the activated prototypes is returned
        # if true, the distance metric for every prototype is returned
        self.Fuzzy = False

    def floatToBits(self, f):
        s = struct.pack('>f', f)
        return hex(struct.unpack('>l', s)[0])

    def computeHamming(self, data, i):
        """Calculate the Hamming distance between two bit strings"""
        prototype = self.prototypes[i]
        count = 0
        for j in range(self.dimensions):
            z = int(self.floatToBits(data[j]), 16) & int(prototype[j], 16)
            while z:
                count += 1
                z &= z - 1  # magic!
        return count

    # The function to get the features for the observation 'data'
    # the argument 'update' is a boolean which indicates whether the representation should
    # check for an update condition (such as meeting the minimal amount of prototype visits).
    # This is useful for debugging
    def GetFeatures(self, data, update):
        if self.distanceMeasure == 'euclidian':
            # tempArr = np.array([1 if np.linalg.norm(data - self.prototypes[i]) < self.threshold else 0 for i in range(len(self.prototypes))])

            if self.updateFunc == 0:  # XGame Paper

                tempArr = np.array(
                    [[i, np.linalg.norm(data - self.prototypes[i])] for i in range(len(self.prototypes))])

                closestPrototypesIndxs = [int(x[0]) for x in sorted(tempArr, key=lambda x: x[1])[:self.numClosest]]

                if update:

                    print('Updating XGame')
                    for i in closestPrototypesIndxs:
                        self.visitCounts[i] += 1

                    if self.beenAroundTheBlock == False:  # use this so we dont have to calculated the max every time
                        maxVisit = max(self.visitCounts)
                        print('Max visit: ' + str(maxVisit))
                        if maxVisit > self.minNumberVisited:
                            self.beenAroundTheBlock = True

                    if self.beenAroundTheBlock:
                        self.updatePrototypesXGame()

            elif self.updateFunc == 1:  # Case Studies

                closestPrototypesIndxs = []
                data = np.array(data)
                for prototype in range(self.numPrototypes):
                    diffArr = abs(data - self.prototypes[prototype])
                    # closestPrototypesIndxs.append(min([1 - diff/self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr]))
                    u = min(
                        [1 - diff / self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr])
                    if u > 0:
                        closestPrototypesIndxs.append(prototype)

                if update:
                    print('Updating Case Studies')
                    # if len(closestPrototypesIndxs) < self.caseStudyN:
                    # 	for i in range(self.caseStudyN - len(closestPrototypesIndxs)):

            return closestPrototypesIndxs

        elif self.distanceMeasure == 'hamming':
            # fuzzy
            # return np.array([self.computeHamming(data,i)/self.threshold for i in range(len(self.prototypes))])
            tempArr = np.array(
                [1 if self.computeHamming(data, i) < self.threshold else 0 for i in range(len(self.prototypes))])

            return np.where(tempArr == 1)[0]

    # the update algorithm defined in the XGame paper
    def updatePrototypesXGame(self):
        self.updatedPrototypes = []
        mostVisitedPrototypeIndexs = [i[0] for i in sorted(enumerate(self.visitCounts), key=lambda x: x[1])]
        count = 0
        for prototype in range(self.numPrototypes):
            if math.exp(-self.visitCounts[prototype]) > random.random():  # remove with probability e^-m (Equation 4)
                self.visitCounts[prototype] = 0
                replacementPrototypeIndex = mostVisitedPrototypeIndexs[-(count + 1)]
                self.prototypes[prototype] = self.prototypes[replacementPrototypeIndex]  # add another prototype

                for dimension in range(self.dimensions):
                    randOffset = (random.random() - .5) / (self.numPrototypes ^ -self.dimensions)
                    self.prototypes[prototype][dimension] += randOffset  # change every dimension to something close by

                self.updatedPrototypes.append([prototype, self.prototypes[prototype], replacementPrototypeIndex])
                count += 1

        self.visitCounts = np.zeros(self.numPrototypes)
        self.beenAroundTheBlock = False

        print('Done updatedPrototypes: updatedPrototypes: ' + str(self.updatedPrototypes))


class BaseKanervaCoder:
    def __init__(self, _startingPrototypes, _dimensions, _numActiveFeatures):
        self.numPrototypes = _startingPrototypes
        self.dimensions = _dimensions
        self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
        self.F = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
        self.sorted_prototype_diffs_indexs = np.zeros(self.numPrototypes)
        self.g = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])

        self.numActiveFeatures = _numActiveFeatures

    def get_features(self, data):
        closestPrototypesIndxs = np.zeros(self.numPrototypes)
        diffs = np.zeros(self.numPrototypes)
        for i in range(self.numPrototypes):
            diffs[i] = np.linalg.norm(data - self.prototypes[i])

        self.sorted_prototype_diffs_indexs = np.argsort(diffs)

        diffs = self.sorted_prototype_diffs_indexs[:self.numActiveFeatures]
        closestPrototypesIndxs[diffs] = 1

        return closestPrototypesIndxs

    def calculate_f(self, data):

        sigmoid = self.sorted_prototype_diffs_indexs / self.numPrototypes
        tempF = np.array([np.random.rand(self.numPrototypes) for i in range(self.dimensions)])
        for i in range(self.dimensions):
            tempF[i] = sigmoid*(1-sigmoid)*data[i] # Setting the whole array here instead of one element at a time
        self.F = tempF.T

    def update_prototypes(self, obs, alpha, delta, phi, th):

        self.calculate_f(obs)
        partA = self.g * np.ones(self.g.shape) - (self.g.T * (alpha * phi)).T
        partB = self.F.T * (1 + alpha * (np.ones(th.shape)*delta - th))

        self.g = partA + partB.T

        partA = alpha*delta*phi
        tempPrototypes = np.array([np.random.rand(self.numPrototypes) for i in range(self.dimensions)])
        tempG = self.g.T

        for i in range(self.dimensions):
            tempPrototypes[i] += partA*tempG[i]


class KrisKanervaCoder:
    def __init__(self, dims, ptypes, n_active, mask=None, limits=None, bias=True, dist=lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
                 seed=None):
        np.random.seed(seed)
        self._n_pts = ptypes
        self._k = n_active
        self.numActiveFeatures = self._k
        self.dimensions = dims
        self.numPrototypes = ptypes
        self.mem_size = self.numPrototypes
        if limits is None:
            limits = np.zeros((dims, 2))
            limits[:, 1] = 1
        self.bias = bias
        self.mask = mask
        self._lims = np.array(limits)
        self._ranges = self._lims[:, 1] - self._lims[:, 0]
        self._pts = np.random.random([self._n_pts, dims])
        self._dist = dist

    @property
    def n_ptypes(self):
        return self._n_pts

    def get_num_active(self):
        return self._k

    def get_features(self, x, **vals):
        if self.mask is None:
            x = x[self.mask]
        xs = (x - self._lims[:, 0]) / self._ranges
        phi = np.zeros(self._n_pts)
        if self.bias:
            phi[np.argpartition(self._dist(self._pts-1, xs), self._k)[:self._k]] = 1
            phi[len(phi)-1] = 1
        else:
            phi[np.argpartition(self._dist(self._pts, xs), self._k)[:self._k]] = 1
        return phi


if __name__ == "__main__":
    rewards =[]
    ranges = np.zeros((4,2))
    ranges[:,1] = 1

    kris = KrisKanervaCoder(4,3000,15,ranges)
    s = time.time()
    for i in range(300):
        x = np.random.rand(4)
    print("kris: %f" % (time.time() - s))

    # jaden = BaseKanervaCoder(3000,4,15)
    # s = time.time()
    # for i in range(300):
    #     x = np.random.rand(4)
    #     jaden.get_features(x)
    # print("jaden: %f" % (time.time() - s))

    s = time.time()
    for i in range(300):
        x = np.random.uniform(size=(2000,300))
        y = np.random.random_integers(0, 1, size=(300))
        np.dot(x,y)
    print("dot: %f" % (time.time() - s))


    for i in range(300):
        x = np.random.uniform(size=(2000,300))
        y = np.random.random_integers(0,1,size=(300))
        x[y].sum
    print("sum: %f" % (time.time() - s))
