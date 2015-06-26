def getExecutorMemory(driver=None):
    from urllib2 import urlopen
    from numpy import array, append, empty, sum
    from os.path import expanduser

    if driver is None:
        fid = open(expanduser('~/spark-master'))
        driver = fid.readline()[8:-6]

    url = urlopen("http://" + driver + ":4040/executors")
    html = url.read()
    
    k = html.find("Memory:")
    k1 = html.find("\n", k)
    k2 = html.find("\n", k1+1)
    used = html[k1:k2].lstrip()
    k3 = html.find("</li>", k2)
    total = html[k2:k3].lstrip()

    print used + " " + total

if __name__ == '__main__':
    getExecutorMemory()
