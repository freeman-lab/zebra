def dirNest(path):
    """
    :param path: string, path to some directory
    :return: dn: list of 2-tuples of strings, the result of recursively splitting the directory structure in path
    """
    from os.path import split, sep
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

        outDir = join(baseDir, 'spark', expDir, expName, chDir, '')

    else:
        expDir = dn[-3][1]
        baseDir = dn[-4][0]
        outDir = join(baseDir, 'spark', expDir, expName, '')

    return outDir


def bz2compress(raw_fname, wipe=False, overwrite=False):
    """
    :param raw_fname: string, full path of file to be compressed
    :param wipe: bool, optional. If set to True, raw file will be deleted after compression. Defaults to False.
    :param overwrite: bool, optional. If set to True, overwrites file sharing name of output file. Defaults to False
    :return:
    """

    import bz2
    import os

    compressed_fname = raw_fname + '.bz2'

    if os.path.exists(compressed_fname) and overwrite is False:
        raise ValueError('File {0} already exists. Call bz2compress with '.format(compressed_fname) +
                         'overwrite=True to overwrite.')

    with open(raw_fname, 'rb') as f:
        data = f.read()

    compressed_file = bz2.BZ2File(compressed_fname, "wb")
    compressed_file.write(data)
    compressed_file.close()

    if wipe:
        os.remove(raw_fname)

    return


def bz2decompress(compressed_fname, wipe=False, overwrite=False):
    """
    :param compressed_fname: string, full path of file to be decompressed
    :param wipe: bool, optional. If set to True, raw file will be deleted after decompression
    :param overwrite: bool, optional. If set to True, overwrites file sharing name of output file. Defaults to False.
    :return:
    """

    import bz2
    import os

    raw_fname = compressed_fname[:-4]  # chopping the '.bz2' extension for raw file name

    if os.path.exists(raw_fname) and overwrite is False:
        raise ValueError('File {0} already exists. Call bz2decompress with '.format(raw_fname) +
                         'overwrite=True to overwrite.')

    infile = bz2.BZ2File(compressed_fname, 'rb')
    data = infile.read()
    infile.close()

    with open(raw_fname, 'wb') as f:
        f.write(data)

    if wipe:
        os.remove(compressed_fname)

    return

def stack_to_tif(stack_path):
    """
    :param stack_path: string, full path of .stack file to be converted to .tif
    :return:
    """

    from os.path import split, sep
    from numpy import fromfile
    import volTools as volt
    from skimage.external import tifffile as tif

    dims = volt.getStackDims(split(stack_path)[0] + sep)

    im = fromfile(stack_path, dtype='int16')
    im = im.reshape(dims[-1::-1])

    tif_path = stack_path.split('.')[0] + '.tif'
    tif.imsave(tif_path, im, compress=1)