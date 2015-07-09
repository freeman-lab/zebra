"""Process electrophysiological recordings of fish behavior and trial structure"""

import numpy as np 

def chopTrials(signal,trialThr=2000):
    """for each unique value in the signal, 
       return the start and stop of each epoch corresponding to that value
    """
    
    allCond = np.unique(signal)        
    chopped = {}
    for c in allCond:
        tmp = np.where(signal == c)[0]
        offs = np.where(np.diff(tmp) > 1)[0]
        offs = np.concatenate((offs, [tmp.size-1]))
        ons = np.concatenate(([0], offs[0:-1] + 1))
        trLens = offs - ons        
        keepTrials = np.where(trLens > trialThr)
        offs = offs[keepTrials]
        ons = ons[keepTrials]
        chopped[c] = (tmp[ons], tmp[offs])
    
    return chopped


def stackInits(frameCh,thrMag=3.8,thrDur=10):    
    """
        Find indices in ephys time corresponding to the onset of each stack in image time
    """
    
    stackInits = np.where(frameCh > thrMag)[0]
    initDiffs = np.where(np.diff(stackInits) > 1)[0]
    initDiffs = np.concatenate(([0], initDiffs+1))    
    stackInits = stackInits[initDiffs]
    keepers = np.concatenate((np.where(np.diff(stackInits) > thrDur)[0], [stackInits.size-1]))
    stackInits = stackInits[keepers]
    
    return stackInits


def getSwims(ch):
    """ Estimate swim timing from ephys recording of motor neurons
    """
    
    from scipy import signal as sig

    # set dead time, in samples
    deadT = 80
    
    fltch = smoothPower(ch)
    peaksT, peaksIndT = getPeaks(fltch, deadT)
    thr = getThreshold(fltch,2600000)
    burstIndT = peaksIndT[np.where(fltch[peaksIndT] > thr[peaksIndT])]
    burstT = np.zeros(fltch.shape)
    burstT[burstIndT] = 1
    
    interSwims = np.diff(burstIndT)
    swimEndIndB = np.where(interSwims > 800)[0]
    swimEndIndB = np.concatenate((swimEndIndB,[burstIndT.size-1]))

    swimStartIndB = swimEndIndB[0:-1] + 1
    swimStartIndB = np.concatenate(([0], swimStartIndB))
    nonShort = np.where(swimEndIndB != swimStartIndB)[0]
    swimStartIndB = swimStartIndB[nonShort]
    swimEndIndB = swimEndIndB[nonShort]

    bursts = np.zeros(fltch.size)
    starts = np.zeros(fltch.size)
    stops = np.zeros(fltch.size)
    bursts[burstIndT] = 1
    starts[burstIndT[swimStartIndB]] = 1
    stops[burstIndT[swimEndIndB]] = 1
    
    return starts, stops, thr


# filter signal, extract power
def smoothPower(ch, kern=None):
    """
    subtract smoothed vector from raw vector and square to estimate swim power
    :param ch:
    :param kern:
    :return:
    """
    from scipy import signal as sig

    if not kern:
        kern = sig.gaussian(121, 20)
        kern = kern/kern.sum()

    smch = np.convolve(ch**2, kern, 'same')
    power = (ch**2 - smch)
    fltch = np.convolve(power, kern, 'same')
    return fltch

# get peaks
def getPeaks(fltch,deadTime=80):
    
    aa = np.diff(fltch)
    peaks = (aa[0:-1] > 0) * (aa[1:] < 0)
    inds = np.where(peaks)[0]    

    # take the difference between consecutive indices
    dInds = np.diff(inds)
                    
    # find differences greater than deadtime
    toKeep = (dInds > deadTime)    
    
    # only keep the indices corresponding to differences greater than deadT 
    inds[1::] = inds[1::] * toKeep
    inds = inds[inds.nonzero()]
    
    peaks = np.zeros(fltch.size)
    peaks[inds] = 1
    
    return peaks,inds

# find threshold
def getThreshold(fltch,wind=180000,shiftScale=1.6):
    
    th = np.zeros(fltch.shape)
    
    for t in np.arange(0,fltch.size-wind, wind):

        interval = np.arange(t, t+wind)
        sqrFltch = fltch ** .5            
        hist, bins = np.histogram(sqrFltch[interval], 1000)
        mx = np.min(np.where(hist == np.max(hist)))
        mn = np.max(np.where(hist[0:mx] < hist[mx]/200.0))        
        th[t:] = (bins[mx] + shiftScale * (bins[mx] - bins[mn]))**2.0

    return th

def load(inFile):
    """Load 10chFlt data from disk, return as a [channels,samples] sized numpy array
    """
    fd = open(inFile, 'rb')
    data = np.fromfile(file=fd, dtype=np.float32)
    data = data.reshape(data.size/10,10).T
    return data