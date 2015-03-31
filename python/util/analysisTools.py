import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
from IPython import display
from itertools import product
from copy import deepcopy
from collections import Iterable
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import inv
from scipy.interpolate import splev

try:
    from thunder import Series
except:
    print "WARNING: thunder could be be loaded from 'personal'"

#----------------------------------------------------------------------------------------------
# functions for cross-sectional plots of 3d data

def crossSectionPlot(pts, axes=(0,1,2), margin=0.1, limits='None', color='axis3', cmap='Blues_r', negcmap=None,  alpha=None):
    
    # compute max and min values if not already given
    if limits=='None':
        maxVals = np.max(pts,axis=0)
        minVals = np.min(pts,axis=0)
    else:
        maxVals = np.asarray(limits.max)
        minVals = np.asarray(limits.min)
        
    # easier names for which indices serve which purpose
    xInd = axes[0]
    yInd = axes[1]
    if color=='axis3':
        colorInd = axes[2]
        
    # compute axis limits
    offsets = (margin/(1.0-margin))*((maxVals-minVals)/2)
    axisMin = minVals-offsets
    axisMax = maxVals+offsets

    # special case where coloring by z-axis
    if color=='axis3':
        color=pts[:,colorInd]

    # get colors/alphas
    if negcmap is None:
        if type(cmap) is np.ndarray:
            c = np.array([cmap[x] for x in color]) 
        else:
            cmap = get_cmap(cmap)
            norm = Normalize(min(color), max(color))
            c = cmap(norm(color))
    else:
        cmapPos, cmapNeg  = get_cmap(cmap), get_cmap(negcmap)
        pos, neg = np.where(color>=0)[0], np.where(color<0)[0]
        colorPos, colorNeg = color[pos], -color[neg]
        c = np.zeros((color.shape[0],4))
        if pos.size != 0:
            normPos = Normalize(0, max(colorPos))
            c[pos,:] = cmapPos(normPos(colorPos))
        if neg.size != 0:
            normNeg = Normalize(0, max(colorNeg))
            c[neg,:] = cmapNeg(normNeg(colorNeg))
    if alpha is not None:
        c[:,3] = alpha 
    
    # generate plot
    plt.scatter(pts[:,xInd], pts[:,yInd], c=c, lw=0)
    plt.xlim([minVals[xInd]-offsets[xInd],maxVals[xInd]+offsets[xInd]])
    plt.ylim([minVals[yInd]-offsets[yInd],maxVals[yInd]+offsets[yInd]])


def crossSectionFigure(pts, margin=0.1, inset='None', limits='None', color='axis3', cmap='Blues_r', negcmap=None):
        
    # plot the x-y view (top-down)
    plt.subplot2grid((3,5),(0,0),rowspan=2,colspan=4)
    crossSectionPlot(pts, axes=(0,1,2), margin=margin, limits=limits, color=color, cmap=cmap, negcmap=negcmap)

    # plot the y-z view (head-on)
    plt.subplot2grid((3,5),(0,4),rowspan=2,colspan=1)
    crossSectionPlot(pts, axes=(2,1,2), margin=margin, limits=limits, color=color, cmap=cmap, negcmap=negcmap)
    
    # plot the x-z view (side-on)
    plt.subplot2grid((3,5),(2,0),rowspan=1,colspan=4)
    crossSectionPlot(pts, axes=(0,2,2), margin=margin, limits=limits, color=color, cmap=cmap, negcmap=negcmap)

    if inset == 'None':
        pass    
    elif type(inset) is np.ndarray:
        plt.subplot2grid((3,5),(2,4))
        plt.plot(inset)
    else:
        pass
        #error msg here
    

def crossSectionSelect(pts, labels, inset='None', margin=0.1, limits='None', color='axis3', cmap='Blues_r', negcmap=None):
    
    #compute the number of clusters
    minLabel = np.min(labels)
    maxLabel = np.max(labels)
    nClusters = maxLabel - minLabel + 1

    fig = plt.figure(figsize=(20,10))

    #start with smallest label and loop
    i = minLabel 
    while 1:
        if i == -1:
            break
        fig.clf()
    
        crossSectionFigure(pts[labels==i,:], margin=margin, limits=limits, color=color, cmap=cmap, negcmap=negcmap)
        if ( type(inset) is np.ndarray ) and ( len(inset.shape) == 2 ) and ( inset.shape[0] == max(labels)+1 ):
            plt.subplot2grid((3,5),(2,4))
            otherIdx = np.hstack((np.arange(i),np.arange(i+1,nClusters))).astype('int')
            plt.plot(inset[otherIdx,:].T, color='black', alpha=0.1)
            plt.plot(inset[i,:].T, color='blue')
        
        display.display(fig)
        display.clear_output(wait=True)
    
        while 1:
            try:
                selection = int(raw_input("Cluster # (-1 to exit):"))
                success = True
            except:
                success = False
            if success==True and (selection==-1 or (selection>=minLabel and selection<=maxLabel)):
                i = selection
                break
            else:
                continue

# #-------------------------------------------------------------------------------------------
# class for linear regression

class RegressionModel():

    def __init__(self, type='ols', constant=True, regularization=None, regScale=None):
        self.type = type
        self.constant = constant 
        self.betas = None

    def fit(self, X, y):

        X = np.array(X, ndmin=2)
        if X.shape[0] == 1:
            X = X.T

        if self.constant:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        if self.type == 'ols':
            self.betas = np.dot(self.pinv(X), y)
        else:
            print "regression type ({}) not supported".format(self.type)

        self.stats = self.predictWithStats(X, y)
        return self

    def predict(self, X):
        if self.betas is None:
            print "Must fit before you can predict"
        else:
            return np.dot(X, self.betas)

    def predictWithStats(self, X, y):
        if self.betas is None:
            print "Must fit before you can predictWithStats"
        else:
            yhat = np.dot(X, self.betas)
            residuals = y - yhat
            SST = np.sum(np.square(y - np.mean(y)))
            SSE = np.sum(np.square(y - yhat))
            rSq = 1 - 1.0*SSE/SST
            return yhat, residuals, rSq

    @classmethod
    def pinv(cls, X):
        return np.dot(inv(np.dot(X.T, X)), X.T)

##--------------------------------------------------------------------------------------------
# class for spline regression

class RegressionSpline():

    def __init__(self, order, nknots, placement='percentile', regularization=None, monotone=False, ext=1):
        self.n = nknots
        self.k = order
        self.placement = placement
        self.regularization = regularization
        self.monotone = monotone
        self.ext = ext
        self.a = None

    def fit(self, x, y):

        if self.placement == 'percentile':
            self.knots = self.getKnotsByPercentile(x, self.n)
        elif self.placement == 'equal':
            self.knots = self.getKnotsEqual(x, self.n)
        else:
            raise ValueError('placement option "{}" not recognized'.format(str(placement)))

        self.knots = self.augmentKnots(self.knots, self.k)

        m = self.n - 2
        nBSplines = self.k + m

        B = []
        for i in xrange(nBSplines):
            w = len(self.knots)*[0]
            w[i] = 1
            B.append(splev(x, [self.knots, w, self.k-1], ext=1))
        B = np.array(B).T

        if self.regularization is not None:
            D = []
            for j in xrange(2, nBSplines):
                c1 = 1.0/(self.knots[j+self.k-2] - self.knots[j-1])
                c2 = 1.0/(self.knots[j+self.k-1] - self.knots[j])
                c3 = 1.0/(self.knots[j+self.k-2] - self.knots[j])
                v = c3*np.array([c1, -(c1 + c2), c2])
                D.append(np.concatenate([np.zeros(j-2), v, np.zeros(nBSplines-j-1)]))
            D = np.array(D)
            B = np.vstack([B, np.sqrt(self.regularization)*D])
            y = np.concatenate([y, np.zeros(nBSplines-2)])
        
        self.a = np.dot(np.dot(inv(np.dot(B.T, B)), B.T), y)    
        return self

    def predict(self, x):
        if self.a is None:
            print "Must fit before you can predict"
        else:
            return splev(x, [self.knots, self.a, self.k-1], ext=self.ext)

    @classmethod
    def getKnotsByPercentile(cls, data, nknots):
        return np.percentile(data, 100*np.linspace(0, 1, num=nknots))

    @classmethod
    def getKnotsEqual(cls, data, nknots):
        return np.linspace(min(x), max(x), num=nknots)

    @classmethod
    def augmentKnots(cls, knots, k):
        return np.concatenate([np.repeat(knots[0], k-1), knots, np.repeat(knots[-1], k-1)])


# #-------------------------------------------------------------------------------------------
# functions to fit a linear kernel
# TODO:
# - add option to not use an instantaneous term in the kernels
# - extend to handle multiple kernels with different maxLag, maxLead, and/or instantaneous choice

# new class-based versions for working on non-RDD data
class LModel:

    def __init__(self, maxLag, maxLead=0, padding=False, mask=None, ):
        self.maxLag = maxLag
        self.maxLead = maxLead
        self.padding = padding
        self.mask = mask

    def fit(self, x, y):
        regMat, yMask = self.kerToReg(x, self.maxLag, self.maxLead, self.padding, self.mask)
        yMasked = y[yMask]
        results = RegressionModel().fit(regMat, yMasked)
        self.betas  = results.betas
        self.stats = results.stats
        return self

    @classmethod
    def getPieces(cls, a):
        a = np.insert(a, 0, False)
        a = np.insert(a, a.shape[0], False)
        indStart = np.where(np.logical_and(np.logical_not(a[:-1]), a[1:]))[0]
        indStop = np.where(np.logical_and(a[:-1], np.logical_not(a[1:])))[0]
        return [np.arange(indStart[i], indStop[i]) for i in xrange(indStart.shape[0])]

    @classmethod
    def kerToReg1(cls, x, maxLag, maxLead=0, padding=False, mask=None):

        N = x.shape[0]

        # default mask uses all values
        if mask is None:
            mask = np.ones(N).astype('bool')

        # find continuous pieces within mask
        inds = getPieces(mask)
        nPieces = len(inds)
        xPieces = [x[inds[i]] for i in xrange(nPieces)]

        # if padding, add zeros for computing estimates near the ends
        if padding:
            leftPad = np.zeros(maxLag)
            rightPad = np.zeros(maxLead)
            xPieces = [np.concatenate((leftPad, piece, rightPad)) for piece in xPieces]

        # create regression matrix
        kerLen = maxLag + 1 + maxLead
        nvals = np.array([piece.shape[0]-kerLen+1 for piece in xPieces])
        mat = np.array([np.concatenate([xPieces[j][i:i+nvals[j]] for j in xrange(nPieces)]) for i in xrange(kerLen)])

        # create mask for y-values that will be used in fitting the model
        # if not using padding, get restriced output values
        yMask = np.zeros(N).astype('bool')
        if not padding:
            yInds = [idxs[maxLag:idxs.shape[0]-maxLead] for idxs in inds]
        else:
            yInds = inds
        yMask[np.concatenate(yInds)] = True

        return mat.T, yMask

    @classmethod
    def kerToReg(cls, x, maxLag, maxLead=0, padding=False, mask=None):

        x = np.array(x, ndmin=2)

        nkers = x.shape[0]
        regMats, yMasks = zip(*[cls.kerToReg1(x[i], maxLag, maxLead, padding, mask) for i in xrange(nkers)])

        return np.hstack(regMats), yMasks[0]

# ---------
# Older function-based version for RDDs

# get continuous pieces of a series as denoted by a boolean mask
def getPieces(a):
    a = np.insert(a, 0, False)
    a = np.insert(a, a.shape[0], False)
    indStart = np.where(np.logical_and(np.logical_not(a[:-1]), a[1:]))[0]
    indStop = np.where(np.logical_and(a[:-1], np.logical_not(a[1:])))[0]
    #return indStart, indStop
    return [np.arange(indStart[i], indStop[i]) for i in xrange(indStart.shape[0])]

# cast kernel fitting for a single explanatory variable into a regression problem
def kerToReg1(x, maxLag, maxLead=0, padding=False, mask=None):

    N = x.shape[0]

    # default mask uses all values
    if mask is None:
        mask = np.ones(N).astype('bool')

    # find continuous pieces within mask
    #indStart, indStop = getPieces(mask)
    inds = getPieces(mask)
    nPieces = len(inds)
    xPieces = [x[inds[i]] for i in xrange(nPieces)]

    # if padding, add zeros for computating esitmates near the ends
    if padding:
        leftPad = np.zeros(maxLag) 
        rightPad = np.zeros(maxLead)
        xPieces = [np.concatenate((leftPad, piece, rightPad)) for piece in xPieces]
    
    # create regression matrix
    kerLen = maxLag + 1 + maxLead
    nvals = np.array([piece.shape[0]-kerLen+1 for piece in xPieces])
    mat = np.array([np.concatenate([xPieces[j][i:i+nvals[j]] for j in xrange(nPieces)]) for i in xrange(kerLen)])
    
    # create mask for y-values that will be used in fitting the model
    # if not using padding, get restriced output values
    yMask = np.zeros(N).astype('bool')
    if not padding:
        yInds = [idxs[maxLag:idxs.shape[0]-maxLead] for idxs in inds]
        #yOut = np.concatenate([piece[maxLag:piece.shape[0]-maxLead] for piece in yPieces])
    else:
        yInds = inds
        #yOut = np.concatenate([piece for piece in yPieces])
    yMask[np.concatenate(yInds)] = True

    return mat, yMask

# cast kernel fitting into a regression problem
def kerToReg(x, maxLag, maxLead=0, padding=False, mask=None):

    x = np.array(x, ndmin=2)

    nkers = x.shape[0]
    regMats, yMasks = zip(*[kerToReg1(x[i], maxLag, maxLead, padding, mask) for i in xrange(nkers)])

    return np.vstack(regMats), yMasks[0]

# fit a linear kernel model
def fitKernel(x, y, maxLag, maxLead=0, padding=False, mask=None): 
    
    regMat, yMask = kerToReg(x, maxLag, maxLead, padding, mask)
    model = thunder.RegressionModel.load(regMat, "linear")
    #TODO: when Series.selectByIndex is available, it should be used here instead of Series.applyValues
    yMasked = y.applyValues(lambda v: v[yMask])
    return model.fit(yMasked)

#-----------------------------------------------------------------------------
# functions to get all records with a given kmeans label

# WARNING:: this function will cache the results, which effectively take up as much room as the original Series again!
def getAllByLabel(data, labels):
    minLabel = labels.min()
    maxLabel = labels.max()
    joined = data.rdd.join(labels.rdd)
    l = [Series(joined.filter(lambda (k,v):v[1]==i).map(lambda (k,v):(k,v[0]))) for i in np.arange(minLabel,maxLabel+1)]
    for i in np.arange(len(l)):
        l[i].rdd.persist()
    return l

# WARNING: this function will cache the results, which will take up as much memory as the number of records with the given label
def getByLabel(data,labels,i):
    joined = data.rdd.join(labels.rdd)
    return Series(joined.filter(lambda (k,v):v[1]==i).map(lambda (k,v):(k,v[0])))

def meanReduceFun(v1,v2):
    num = v1[1]*v1[0] + v2[1]*v2[0]
    denom = v1[1] + v2[1]
    return (1.0*num/denom, denom)

def getAllMeansByLabel(data,labels):
    return Series(data.rdd.join(labels.rdd).map(lambda (k,v):(v[1],(v[0],1))).reduceByKey(meanReduceFun).map(lambda (k,v):(k,v[0])))

def reduceByLabel(data,labels,function):
    return Series(data.rdd.join(labels.rdd).map(lambda (k,v):(v[1],v[0])).reduceByKey(function))

#------------------------------------------
# functions for combining data across trials

# reshape a time series into a matrix with one trial per line
def reshapeByTrial(timeSeries,trialLen):
    L = len(timeSeries)
    nTrials = L/trialLen

    # remove an partial trial at the end of the time series
    timeSeries = timeSeries[:-(L%trialLen)]
    
    # reshape as matrix
    return np.reshape(timeSeries,(nTrials,trialLen))
    
# combine data across trials through a given function (e.g. mean,std)
def applyAcrossTrials(f,timeSeries,trialLen):
    return np.apply_along_axis(f,0,reshapeByTrial(timeSeries,trialLen))

# utility functions for common ways of combining data across trials
def trialAverage(timeSeries,trialLen):
    return applyAcrossTrials(np.mean,timeSeries,trialLen)

def trialStd(timeSeries,trialLen):
    return applyAcrossTrials(np.std,timeSeries,trialLen)


#------------------------------------------------------------
# functions for "spotlight" (i.e. nearest-neighbor) analyses

def reshapeByNN(series, k):

    keys = series.keys().collect()
    key_array = np.array(keys)
    dist, idx = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto').fit(key_array).kneighbors(key_array)
    
    nbr = [ [] for x in xrange(idx.shape[0])]
    it = np.nditer(idx, flags=['multi_index'])
    for x in it:
        nbr[x].append(it.multi_index[0])

    def f(x):
        k,v = x[0], x[1]
        i = keys.index(k)
        return [(n, v) for n in nbr[i]]

    return series.rdd.flatMap(f).groupByKey().map(lambda (k,v): (keys[k], np.array(v.data)))

def meanNN(series, k):
    return Series(reshapeByNN(series, k).mapValues(lambda v: np.mean(v, axis=0)), index=series.index).__finalize__(series, noPropagate=('dype', 'index'))
