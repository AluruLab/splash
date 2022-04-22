#!python3
#
# Copyright 2020 Georgia Tech Research Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Author(s): Tony C. Pan
#

import numpy as np
import pandas as pd
import argparse

from matrix_io import read_exp, read_csv, write_csv



def main():
    parser = argparse.ArgumentParser(description='matrix comparisons')
    parser.add_argument("precision", action='store', default='single',
                        choices=['single', 'double', 'half', 'pearson', 'spearman', 'kendall', 'distance', 'distance3'],
                        help='precision level, default = pearson')
    parser.add_argument('-f', '--first', action='store',
                        help='first matrix', required=True)
    parser.add_argument('-s', '--second', action='store',
                        help='second matrix', required=True)
    parser.add_argument('-o', '--output', action='store', default=None,
                        help='if specified, output per element comparison.  else report if all are equal.')
    parser.add_argument('-d', '--double', type=int, action='store', default=0,
                        help='double precision or single precision comparison')
    parser.add_argument('-m', '--maxdiff', type=int, action='store', default=1,
                        help='compute the maximum difference')


    args = parser.parse_args()

    first = args.first
    second = args.second
    output_fn = args.output
    if args.double == 0:
        datatype = np.float32
    else:
        datatype = np.float64


    rtol = 1e-05
    if args.precision in ['low']:
        atol = 1e-02
    elif args.precision in ['half']:
        atol = 1e-04
    elif args.precision in ['single']:
        atol = 1e-08
    elif args.precision in ['double']:
        atol = 1e-16
    elif args.precision in ['pearson']:
        atol = 1e-07
    elif args.precision in ['spearman']:
        atol = 1e-02
    elif args.precision in ['kendall']:
        atol = 1e-08
    elif args.precision in ['distance']:
        atol = 1e-08
    elif args.precision in ['distance3']:
        atol = 1e-03
    else:
        atol = None

    f = None
    if first.endswith('.exp'):
        f = read_exp(first, dtype=datatype).to_numpy()
    elif first.endswith('csv'):
        f = read_csv(first, dtype=datatype).to_numpy()

    s = None
    if second.endswith('.exp'):
        s = read_exp(second, dtype=datatype).to_numpy()
    elif second.endswith('csv'):
        s = read_csv(second, dtype=datatype).to_numpy()

    if args.maxdiff == 1:
        absdiff = np.absolute(f - s)
        maxdiff = np.amax(absdiff)
        meandiff = np.mean(absdiff)
        stdevdiff = np.std(absdiff, ddof=1)

        print("absolute value difference max {}, mean {}, stdev {}".format(maxdiff, meandiff, stdevdiff))
    
        posmax = np.argmax(absdiff)
        ind = np.unravel_index(posmax, absdiff.shape)
        print("absolute value difference max pos {}, first {}, second {}".format(ind, f[ind], s[ind]))
    else:
        # spearman at atol 1e-3, pearson at atol 1e-7
        print("relative tolerance {}, absolute tolerance {}".format(rtol, atol))
        if output_fn is None:
            result = np.allclose(f, s, equal_nan=True, rtol=rtol, atol=atol)
            print("All elements are equal: {}".format(result))

        else:
            result = np.isclose(f, s, rtol=rtol, atol=atol, equal_nan=True)
            df = pd.DataFrame(data=result)

            write_csv(df, output_fn, header=True, index=True)

            (rr, cc) = np.where(result == False)
            print("unequal at: ")
            coords = list(zip(rr, cc))
            for c in coords:
                print("(r, c) {}: first: {}, second {}".format(c, f[c], s[c]))
        
    
if __name__ == '__main__':
    main()