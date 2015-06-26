def dirNest(path):
    """
    :param path: string, path to some directory
    :return: dn: list of 2-tuples of strings, the result of recursively splitting the directory structure in path
    """
    from os.path import split
    dn = []
    while path is not sep:
        tmp = split(path)
        path = tmp[0]
        dn.append(tmp)
    return dn[-1::-1]

def dirStructure(rawDir):
    """
    :param rawDir: string, path to raw data
    :return: regDir, serDir, matDir: strings, paths for saved data
    """
    from os.path import join
    dn = dirNest(rawDir)

    expName = dn[-2][1]

    # Check whether this is a multicolor experiment
    if expName[0:2] == 'CH':
        expName = dn[-3][1]
        expDir = dn[-4][1]
        chDir = dn[-2][1]
        baseDir = dn[-5][0]

        regDir = join(baseDir, 'spark', expDir, expName, chDir, '', 'reg', '')
        serDir = join(baseDir, 'spark', expDir, expName, chDir, '', 'series', '')
        matDir = join(baseDir, 'spark', expDir, expName, chDir, '', 'mat', '')

    else:
        expName = dn[-2][1]
        expDir = dn[-3][1]
        baseDir = dn[-4][0]

        regDir = join(baseDir, 'spark', expDir, expName, '', 'reg', '')
        serDir = join(baseDir, 'spark', expDir, expName, '', 'series', '')
        matDir = join(baseDir, 'spark', expDir, expName, '', 'mat', '')

    return regDir, serDir, matDir
