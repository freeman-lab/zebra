""" file i/o tools for analyzing light sheet data"""

def getStackDims(inDir):
    """parse xml file to get dimension information of experiment. 
    Returns [x,y,z] dimensions as a list of ints 
    """
    import xml.etree.ElementTree as ET 
    
    dims = ET.parse(inDir+'ch0.xml')
    root = dims.getroot()

    for info in root.findall('info'):
        if info.get('dimensions'):
            dims = info.get('dimensions')
        
    dims = dims.split('x')
    dims = [int(float(num)) for num in dims]
    
    return dims

def getStackFreq(inDir):
    """Get the temporal data from the Stack_frequency.txt file found in 
    directory inDir. Return volumetric sampling rate in Hz, 
    total recording length in S, and total number
    of planes in a tuple.
    """
    f = open(inDir + 'Stack_frequency.txt')    
    times = [float(line) for line in f]
    
    # third value should be an integer
    times[2] = int(times[2])
    
    return times    
    
