def getClusterStatus(driver=None):
    from urllib2 import urlopen
    from numpy import array, append, empty, sum
    from os.path import expanduser

    if driver is None:
        fid = open(expanduser('~/spark-master'))
        driver = 'http://' + fid.readline()[8:-6] + ":8080"
    else:
        driver = 'http://' + driver + ":8080"

    url = urlopen(driver)
    html = url.read()

    # find executor hostnames and ports
    idx = array([0])
    while True:
        idx = append(idx, html.find('int.janelia.org-', idx[-1]+1))
        if idx[-1] == -1:
            break
    idx = idx[1:-1]
    hostnames = empty(idx.shape[0], dtype='S100')
    for i in xrange(idx.shape[0]):
        h = html[idx[i]-7:idx[i]-1]
        p = html[idx[i]+16:idx[i]+21]
        hostnames[i] = h + ".int.janelia.org:" + p

    # get data 
    status = empty(idx.shape[0], dtype='S100')
    memory = empty(idx.shape[0], dtype='S100')
    for i in xrange(idx.shape[0]):
        k = html.find(hostnames[i] + "</td>")
        k1 = html.find("<td>", k+35)
        k2 = html.find("</td>", k+35)
        kstart = k1+4
        kstop = k2
        status[i] = html[kstart:kstop]
        k1 = html.find('<td>', k2)
        k2 = html.find('</td>', k1)
        kstart = k1+4
        kstop = k2
        memory[i] = html[kstart:kstop]

    # compute total memory
    total = empty(idx.shape[0], dtype='float')
    used = empty(idx.shape[0], dtype='float')
    for i in xrange(idx.shape[0]):
        k1 = memory[i].find(" ")
        total[i] =  memory[i][:k1] 
        k2 = memory[i].find("(", k1)
        k3 = memory[i].find(" ", k2)
        used[i] = float(memory[i][k2+1:k3])
    totalUsed = sum(used)
    totalAvailable= sum(total)
    perc = 100.0*totalUsed/totalAvailable

    # print results
    for i in xrange (idx.shape[0]):
        print hostnames[i] + '\t' + status[i] + '\t' + memory[i]
    print 'total used: {} out of {} ({}%)'.format(totalUsed, totalAvailable, perc)
    return status

if __name__ == '__main__':
    getExecutorStatus()
