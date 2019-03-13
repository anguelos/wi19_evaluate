import numpy as np

def get_d_plus(D):
    all_vals=np.sort(np.unique(D))
    return np.min(all_vals[1:]-all_vals[:-1])/2


def get_map(D,classes):
    correct_retrievals = classes[None,:]==classes[:,None]
    Dp = D+get_d_plus(D)*correct_retrievals
    sorted_indexes=np.argsort(Dp,axis=1)

    assert np.all(sorted_indexes[:,0]==np.arange(sorted_indexes.shape[0])) # TODO(anguelos) remove samity check
    non_singleton_idx = correct_retrievals.sum(axis=1) > 1

    #removing singletons as queries
    correct_retrievals=correct_retrievals[non_singleton_idx,:]
    sorted_indexes = sorted_indexes[non_singleton_idx, :]

    sorted_indexes=sorted_indexes[:,1:] # removing self
    sorted_retrievals = correct_retrievals[np.arange(sorted_indexes.shape[0],dtype="int64")[:, None], sorted_indexes]
    sorted_retrievals=sorted_retrievals[:,1:]
    max_precision=np.cumsum(np.ones_like(sorted_retrievals),axis=1)
    max_precision=np.minimum(max_precision,sorted_retrievals.sum(axis=1)[:,None])

    P_at=np.cumsum(sorted_retrievals,axis=1).astype("float")/max_precision
    AP=(P_at*sorted_retrievals)/sorted_retrievals.sum(axis=1)[:,None]
    mAP = AP.mean()
    return mAP

