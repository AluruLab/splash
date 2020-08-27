#!python3

import pandas as pd

import numpy as np

# read input data
def read_exp(filename, nVars=None, nSamples=None, header=0, index=0, delimiter='\t', skip=False, dtype=np.float64):
    # defaults to using top row as headers and first column as row names.
    if skip:
        if nVars is None or nVars < 1:
            if nSamples is None or nSamples < 1:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, skiprows = [1,2])
            else:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, skiprows = [1,2], usecols=list(range(nSamples + 2)))
        else:
            if nSamples is None or nSamples < 1:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, skiprows = [1,2], nrows=nVars)
            else:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, skiprows = [1,2], nrows=nVars, usecols=list(range(nSamples + 2)))
    else:
        if nVars is None or nVars < 1:
            if nSamples is None or nSamples < 1:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index)
            else:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, usecols=list(range(nSamples + 2)))
        else:
            if nSamples is None or nSamples < 1:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, nrows=nVars)
            else:
                data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, nrows=nVars, usecols=list(range(nSamples + 2)))

    # print(data)
    data = data.drop(columns=['Alias'])
    return data.astype(dtype)


# read input data
def read_csv(filename, nVars=None, nSamples=None, header=0, index=0, delimiter=',', dtype=np.float64):
    # defaults to using top row as headers and first column as row names.
    if nVars is None or nVars < 1:
        if nSamples is None or nSamples < 1:
            data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index)
        else:
            data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, usecols=list(range(nSamples)))
    else:
        if nSamples is None or nSamples < 1:
            data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, nrows=nVars)
        else:
            data = pd.read_csv(filename, sep=delimiter, header=header, index_col=index, nrows=nVars, usecols=list(range(nSamples)))

    # print(data)
    # print("read data, dim {} ".format(data.shape))
    return data.astype(dtype)


# write input data
def write_csv(df, filename, header=True, index=True):
    df.to_csv(filename, sep=',', header=header, index=index)
    pass


# write input data.  genes in rows, samples in columns.
def write_exp(df, filename):
    df2 = df
    df2.insert(0, 'Alias', '---')
    df2.to_csv(filename, sep='\t', header=True, index=True)
    pass
