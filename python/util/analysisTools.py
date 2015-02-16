import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
from IPython import display
from itertools import product
from copy import deepcopy
from collections import Iterable
from sklearn.neighbors import NearestNeighbors

try:
    from thunder import RegressionModel, Series
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

#-------------------------------------------------------------------------------------------
# functions to fit a linear kernel

# make a regression matrix with an arbirary number of inputs
def kernelMatrix(timeSeries,maxLag,maxLead=0):
    nInputs = timeSeries.shape[1]
    l = [kernelMatrix1(timeSeries[:,i],maxLag,maxLead) for i in np.arange(nInputs)]
    return np.vstack(l)

# make regression matrix with zero padding to predict first element (no need to change target time-series)
def kernelMatrix1(timeSeries,maxLag,maxLead=0):
    timeSeries = np.squeeze(timeSeries)
    lagPadding = np.squeeze(np.zeros((1,maxLag)))
    leadPadding = np.squeeze(np.zeros((1,maxLag)))
    w = np.hstack((lagPadding,timeSeries,leadPadding))
    n = timeSeries.size
    m = maxLag + 1 + maxLead
    mat = np.zeros((m,n))
    for i in np.arange(0,m):
        mat[i,:] = w[i:i+n]
    return mat

# fit a linear kernel model
def fitKernel(input, response, maxLag, maxLead=0): 
    regressionMat = kernelMatrix(input, maxLag, maxLead)
    model = RegressionModel.load(regressionMat, "linear")   
    return model.fit(response)

# helper function that performs confoluction on a single record
def convolve1(timeSeries,kernel,maxLead=0):
    maxLag = len(kernel) - maxLead - 1
    matrix = kernelMatrix(timeSeries,maxLag,maxLead)
    model =  RegressionModel.load(matrix, "multiply")
    return model.fit(kernel)

#def convolve(data,kernels,maxLead=0)

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

#-------------------------------------------------------------------------------
# functions to deal with TimeSeries with a heirarchical orgainizational structure
# AKA: a slap-dash version of MultiIndexing from Pandas

def makeMultiIndex(array):
    levels = product(*[np.unique(array[i,:]) for i in xrange(len(array))])
    levels = [l for l in levels]
    index = zip(*array)
    return index, levels

def reshapeByIndex(data, index, levels):
    full =  [(np.array([data[x] for x in xrange(len(data)) if index[x]==l]), l) for l in levels]
    array, index = zip(*[x for x in full if x[0].size != 0])
    return array, index

#def aggregrateByGroup(data,index,level,function):
#    array, ind = reshapeByIndex(data, *makeMultiIndex(index[:level+1,:]))
#    return np.apply_along_axis(function, 1, array), np.squeeze(np.array(zip(*ind)))

def aggregrateByGroup(data, index, level, function):
    if type(level) is int or len(level)==1:
        ind = np.array([index[level]])
    else:
        ind = index[level]
    array, inds = reshapeByIndex(data, *makeMultiIndex(ind))
    array = np.array(map(function, array))
    return array, inds

def selectByGroup(data, index, level, val):
    if not isinstance(val, Iterable):
        val = [val]
    if not isinstance(level, Iterable):
        level = [level]

    remove = []
    for i in xrange(len(val)):
        if not isinstance(val[i], Iterable):
            val[i] = [val[i]]
            remove.append(level[i])
    
    p = product(*val)
    s = set([x for x in p])

    array, ind = reshapeByIndex(data, *makeMultiIndex(index))
    ind = np.array(ind)
    array, ind = zip(*[(array[i], ind[i]) for i in xrange(ind.shape[0]) if tuple(ind[i,level]) in s])
    array = np.concatenate(array)
    ind = np.squeeze(np.delete(np.array(ind).T, remove, axis=0))
    return array, ind

    
def selectByGroup2(data,index,level,val):
    if type(val) is int:
        values = set([val])
    else:
        values = set(val)
    array, ind = reshapeByIndex(data, *makeMultiIndex(index[:level+1,:]))
    array, ind =  zip(*[(array[i], ind[i]) for i in xrange(len(array)) if ind[i][-1] in values])
    if type(val) is int or len(val)==1:
        ind = zip(*zip(*ind)[:-1])
    return np.array(array), ind

#------------------------------------------------------------
# functionf for "spotlight" (i.e. nearest-neighbor) analyses

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
