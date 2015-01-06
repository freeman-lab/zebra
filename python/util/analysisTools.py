import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from thunder import RegressionModel, Series

#----------------------------------------------------------------------------------------------
# functions for cross-sectional plots of 3d data

def crossSectionPlot(pts, axes=(0,1,2), margin=0.1, limits='None', color='axis3', cmap='Blues_r'):
    
    # compute max and min values if not already given
    if limits=='None':
        maxVals = np.max(pts,axis=0)
        minVals = np.min(pts,axis=0)
    else:
        maxVals = np.asarray(limits.max)
        minVals = np.asarray(limits.min)
        
    # easier names for which indices server what purpose
    xInd = axes[0]
    yInd = axes[1]
    if color=='axis3':
        colorInd = axes[2]
        
    # compute axis limits
    offsets = (margin/(1.0-margin))*((maxVals-minVals)/2)
    axisMin = minVals-offsets
    axisMax = maxVals+offsets

    # compute colors
    if color=='axis3':
        color=pts[:,colorInd]
    
    # generate plot
    plt.scatter(pts[:,xInd], pts[:,yInd], c=color, cmap=cmap)
    plt.xlim([minVals[xInd]-offsets[xInd],maxVals[xInd]+offsets[xInd]])
    plt.ylim([minVals[yInd]-offsets[yInd],maxVals[yInd]+offsets[yInd]])


def crossSectionFigure(pts, margin=0.1, inset='None', limits='None', color='axis3', cmap='Blues_r'):
        
    # plot the x-y view (top-down)
    plt.subplot2grid((3,5),(0,0),rowspan=2,colspan=4)
    crossSectionPlot(pts, axes=(0,1,2), margin=margin, limits=limits, color=color, cmap=cmap)

    # plot the y-z view (head-on)
    plt.subplot2grid((3,5),(0,4),rowspan=2,colspan=1)
    crossSectionPlot(pts, axes=(2,1,2), margin=margin, limits=limits, color=color, cmap=cmap)
    
    # plot the x-z view (side-on)
    plt.subplot2grid((3,5),(2,0),rowspan=1,colspan=4)
    crossSectionPlot(pts, axes=(0,2,2), margin=margin, limits=limits, color=color, cmap=cmap)

    if inset == 'None':
        pass    
    elif type(inset) is np.ndarray:
        plt.subplot2grid((3,5),(2,4))
        plt.plot(inset)
    else:
        pass
        #error msg here
    

def crossSectionSelect(pts, labels, inset='None', margin=0.1, limits='None', color='axis3', cmap='Blues_r' ):
    
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
    
        crossSectionFigure(pts[labels==i,:], margin=margin, limits=limits, color=color, cmap=cmap)
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
# a class-based implementation of the linear kernel model

