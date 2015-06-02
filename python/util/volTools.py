""" file i/o tools for analyzing light sheet data"""


def getStackDims(inDir, channel=0):
    """
    :param inDir: a string representing a path to a directory containing metadata
    :param channel: an int representing the channel of interest, default is 0)
    :return: dims, a list of integers representing the xyz dimensions of the data
    """
    import xml.etree.ElementTree as ET

    dims = ET.parse(inDir+'ch' + str(channel) + '.xml')
    root = dims.getroot()

    for info in root.findall('info'):
        if info.get('dimensions'):
            dims = info.get('dimensions')

    dims = dims.split('x')
    dims = [int(float(num)) for num in dims]

    return dims


def getStackFreq(inDir):
    """
    Get the temporal data from the Stack_frequency.txt file found in
    directory inDir. Return volumetric sampling rate in Hz,
    total recording length in S, and total number
    of planes in a tuple.
    """
    f = open(inDir + 'Stack_frequency.txt')
    times = [float(line) for line in f]

    # third value should be an integer
    times[2] = int(times[2])

    return times


def getStackData(rawPath, channel=0, frameNo=0):
    """
    :param rawPath: string representing a path to a directory containing raw data
    :param channel: int representing the channel of interest, default is 0
    :param frameNo: int representing the timepoint of the data desired, default is 0
    :return: dims: list of integers representing the xyz dimensions of the data
    """

    from numpy import fromfile
    from string import Template

    dims = getStackDims(rawPath, channel)
    fName = Template('TM${x}_CM0_CHN${y}.stack')
    nDigits_frame = 5
    nDigits_channel = 2
    tmpFName = fName.substitute(x = str(frameNo).zfill(nDigits_frame), y = str(channel).zfill(nDigits_channel))
    im = fromfile(rawPath + tmpFName, dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im

def vidEmbed(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype, and display as HTML5 video.
    Credit: Fernando Perez
    """
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    return HTML(data=video_tag)


def volumeMask(vol):
    """
    :param vol: a 3-dimensional numpy array
    :return: mask, a binary mask with the same shape as vol, and mCoords, a list of (x,y,z) indices representing the
    masked coordinates.
    """
    from numpy import array, where
    from scipy.signal import medfilt2d
    from skimage.filter import threshold_otsu
    from skimage import morphology as morph

    filtVol = array([medfilt2d(x.astype('float32')) for x in vol])

    thr = threshold_otsu(filtVol.ravel())
    mask = filtVol > thr
    strel = morph.selem.disk(3)
    mask = array([morph.binary_closing(x, strel) for x in mask])
    mask = array([morph.binary_opening(x, strel) for x in mask])

    z, y, x = where(mask)
    mCoords = zip(x, y, z)

    return mask, mCoords


def projFigure(vol, limits, plDims=[16,10,5], zscale=5, colors='gray', title=None):
    """
    Display vol.max(dim) - vol.min(dim) for dims in [0,1,2]
    Heavily adapted from Jason Wittenbach's crossSectionPlot. 
    """
    import matplotlib.pyplot as plt

    x, y, z = plDims
    grid = (y+z, x+z)
    zRat = zscale*(float(y)/z)
    plt.figure(figsize=grid[-1::-1])   
    
    # plot the x-y view (top-down)
    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=y, colspan=x)
    plt.imshow(vol.max(0) + vol.min(0), clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')
    ax1.axes.xaxis.set_ticklabels([])
    
    if title:
        plt.title(title)
    
    # plot the x-z view (side-on)
    ax2 = plt.subplot2grid(grid, (y, 0),rowspan=z, colspan=x)
    plt.imshow(vol.max(1)+vol.min(1), aspect=zRat, clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')
    
    # plot the y-z view (head-on)
    ax3 = plt.subplot2grid(grid, (0, x), rowspan=y, colspan=z)
    plt.imshow((vol.max(2)+vol.min(2)).T, aspect=1/zRat, clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')
    ax3.axes.yaxis.set_ticklabels([])