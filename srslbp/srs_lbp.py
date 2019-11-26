#!/usr/bin/env python

import numpy as np
import sys
import sklearn.decomposition
import scipy
import wi19
import time

"""SRS LBP.
This script implements the paper 'Sparse Radial Sampling LBP for Writer Identification'

https://arxiv.org/pdf/1504.06133.pdf
@inproceedings{nicolaou2015sparse,
  title={Sparse radial sampling LBP for writer identification},
  author={Nicolaou, Anguelos and Bagdanov, Andrew D and Liwicki, Marcus and Karatzas, Dimosthenis},
  booktitle={2015 13th International Conference on Document Analysis and Recognition (ICDAR)},
  pages={716--720},
  year={2015},
  organization={IEEE}
}

For more information contact Anguelos dot Nicolaou at gmail dot com.
"""

def read_csv(fname):
    lines=[l.split(",") for l in open(fname).read().strip().split("\n")]
    fnames=[line[0] for line in lines]
    values=[[float(col) for col in line[1:]] for line in lines]
    return np.array(values),np.array(fnames)


def block_normalise(values,block_sizes=None,suppress_bins=[0]):
    if block_sizes is None:
        block_sizes = [256]*(values.shape[1]/256)
    res_values=values.copy()
    for block in range(len(block_sizes)):
        block_start = sum(block_sizes[:block])
        block_end = sum(block_sizes[:block+1])
        for bin in suppress_bins:
            res_values[:,256] =0
            res_values[:,bin+block_start] = 0
        block_sum=res_values[:,block_start:block_end].sum(axis=1)
        res_values[:,block_start:block_end]/=block_sum[:,None]
    return res_values


def hellinger_normalise(values):
    return np.sign(values)*np.abs(values)**.5

def l2_normalise(values):
    return values/((np.sum(values**2,1)**.5)+.00000000000001)[:,None]

def pca_reduce(values,pca_values=None,n_components=200,l1out=False):
    pca = sklearn.decomposition.PCA(copy=False, n_components=n_components)
    if l1out:
        res = np.zeros_like(pca.fit(values).transform(values))
        for k in range(res.shape[0]):
            idx = np.arange(res.shape[0]) != k
            res[k, :] = pca.fit(values[idx, :]).transform(values[~idx, :])
            sys.stderr.write(".")
            sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
        return res

    if pca_values is None or pca_values is values:
        pca_values = values
        print "PCA ... ",
        return pca.fit_transform(values)
        print "done."
        if pca_values.shape[0]>2000:
            pass
            #idx=np.arange(pca_values.shape[0])
            #bp.random.shuffle(idx)
            #pca_values=pca_values[idx[:pca_values.shape[1]],:]

    return pca.fit(pca_values).transform(values)

def print_values_dist(values,fnames,metric="cityblock"):
    compressed_distmat=scipy.spatial.distance.pdist(values,metric=metric)
    D = scipy.spatial.distance.squareform(compressed_distmat)
    csv_rows=[]
    for item in range(fnames.shape[0]):
        columns = ["{}".format(c) for c in D[item,:].tolist()]
        csv_rows.append(fnames[item]+","+",".join(columns))
    return "\n".join(csv_rows)

def print_values_dist(values,fnames,metric="cityblock",fd=None):
    if fd is None:
        fd=sys.stdout
    compressed_distmat=scipy.spatial.distance.pdist(values,metric=metric)
    D = scipy.spatial.distance.squareform(compressed_distmat)
    csv_rows=[]
    for item in range(fnames.shape[0]):
        columns = ["{}".format(c) for c in D[item,:].tolist()]
        fd.write(fnames[item]+","+",".join(columns)+"\n")
    fd.flush()

def pipeline(validation_values,pcaset_values,n_components=200,l1out=False):
    t=time.time()
    #print "Pipeline1: {} msec".format(int(1000.*(time.time()-t)))
    validation_values=block_normalise(validation_values)
    #print "Pipeline2: {} msec".format(int(1000. * (time.time() - t)))
    if pca_values is not None:
        pcaset_values = block_normalise(pcaset_values)
    #print "Pipeline3: {} msec".format(int(1000. * (time.time() - t)))
    validation_values=pca_reduce(validation_values,pca_values=pcaset_values,n_components=n_components,l1out=l1out)
    validation_values=hellinger_normalise(validation_values)
    validation_values=l2_normalise(validation_values)
    return validation_values

if __name__ == "__main__":
    params = {"validation_csv": "", "pca_csv": "{validation_csv}","output":"stdout","nb_components":200,"metric":"cityblock","l1out":0}
    params,_ = wi19.get_arg_switches(params)
    validation_values,validation_fnames=read_csv(params["validation_csv"])
    if params["validation_csv"]!=params["pca_csv"]:
        pca_values,_=read_csv(params["pca_csv"])
    else:
        pca_values=None
    validation_values=pipeline(validation_values,pca_values,l1out=params["l1out"])
    #out_csv=print_values_dist(validation_values,validation_fnames,metric=params["metric"])
    if params["output"]=="stdout":
        fd=sys.stdout
    else:
        fd=open(params["output"],"w")
    print_values_dist(validation_values, validation_fnames, metric=params["metric"],fd=fd)