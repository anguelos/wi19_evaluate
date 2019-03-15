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

    max_P_at=np.cumsum(sorted_retrievals,axis=1).astype("float")/max_precision
    # max_P_at is the # correct retrivals @, divied by # of relevant items.
    AP=(max_P_at*sorted_retrievals)/sorted_retrievals.sum(axis=1)[:,None]
    mAP = AP.mean()
    return mAP


def get_classification_metrics(stop_indexes,D,classes):
    correct_retrievals = classes[None,:]==classes[:,None]
    Dp = D+get_d_plus(D)*correct_retrievals
    sorted_indexes=np.argsort(Dp,axis=1)

    assert np.all(sorted_indexes[:,0]==np.arange(sorted_indexes.shape[0])) # TODO(anguelos) remove samity check

    # not removing singletons as queries

    sorted_indexes=sorted_indexes[:,1:] # removing self
    sorted_retrievals = correct_retrievals[np.arange(sorted_indexes.shape[0],dtype="int64")[:, None], sorted_indexes]
    sorted_retrievals=sorted_retrievals[:,1:]

    relevant_count=correct_retrievals.sum(axis=1).reshape(-1,1)

    P_at=np.cumsum(sorted_retrievals,axis=1).astype("float") / np.cumsum(np.ones_like(sorted_retrievals))
    R_at = np.cumsum(sorted_retrievals, axis=1).astype("float") / relevant_count

    # If there are no relevant items and the user replied none, he gets 100% for the query
    P_at = np.concatenate([relevant_count == 0,P_at],axis=1)
    R_at = np.concatenate([relevant_count == 0, R_at], axis=1)
    # When there are no relevant items, Recall is always 100 %
    R_at[relevant_count == 0,:] = 1

    P_by_query=P_at[:,stop_indexes]
    R_by_query=R_at[:,stop_indexes]
    P = P_by_query.mean()
    R = R_by_query.mean()
    Fm = (2*P*R)/(P+R)
    RoC={"Precision":P_by_query.mean(axis=0),"Recall":R_by_query.mean(axis=0)}
    return Fm,P,R,RoC


