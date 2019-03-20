import numpy as np
from matplotlib import pyplot as plt

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


def _get_sorted_retrievals(D,classes,remove_self_column=True):
    correct_retrievals = classes[None, :] == classes[:, None]
    Dp = D+get_d_plus(D)*correct_retrievals
    sorted_indexes=np.argsort(Dp,axis=1)
    if remove_self_column:
        sorted_indexes=sorted_indexes[:,1:] # removing self
    sorted_retrievals = correct_retrievals[np.arange(sorted_indexes.shape[0], dtype="int64")[:, None], sorted_indexes]
    return sorted_retrievals


def _get_precision_recall_matrices(D, classes, remove_self_column = True):
    sorted_retrievals = _get_sorted_retrievals(D, classes, remove_self_column = remove_self_column)
    relevant_count = sorted_retrievals.sum(axis=1).reshape(-1,1)
    precision_at = np.cumsum(sorted_retrievals,axis=1).astype("float") / np.cumsum(np.ones_like(sorted_retrievals),axis=1)
    recall_at = np.cumsum(sorted_retrievals, axis=1).astype("float") / np.maximum(relevant_count, 1)
    recall_at[relevant_count.reshape(-1) == 0, :] = 1
    return precision_at, recall_at

def _compute_map(precision_at,sorted_retrievals):
    AP = (precision_at*sorted_retrievals).sum(axis=1)/sorted_retrievals.sum(axis=1)
    return AP.mean()

def _compute_fscore(sorted_retrievals,relevant_estimate):
    idx=np.arange(relevant_estimate.size,dtype="int64")
    tp=float(sorted_retrievals.cumsum(axis=1)[idx,relevant_estimate].sum())
    retrieved=relevant_estimate.sum()
    relevant=sorted_retrievals.sum()
    precision=tp/retrieved
    recall=tp/relevant
    fscore = 2*precision*recall/(precision+recall)
    return fscore, precision, recall


def get_classification_metrics(stop_indexes,D,classes,remove_self_column=True):
    P_at, R_at = get_precision_recall_matrices(D, classes, remove_self_column=remove_self_column)
    nb_queries = D.shape[0]
    #correct_retrievals = classes[None,:]==classes[:,None]
    #Dp = D+get_d_plus(D)*correct_retrievals
    #sorted_indexes=np.argsort(Dp,axis=1)

    #assert np.all(sorted_indexes[:,0]==np.arange(sorted_indexes.shape[0])) # TODO(anguelos) remove samity check

    # not removing singletons as queries

    #sorted_indexes=sorted_indexes[:,1:] # removing self
    #print sorted_indexes.shape
    #sorted_retrievals = correct_retrievals[np.arange(sorted_indexes.shape[0],dtype="int64")[:, None], sorted_indexes]
    #sorted_retrievals=sorted_retrievals[:,1:]

    #relevant_count=sorted_retrievals.sum(axis=1).reshape(-1,1)

    #P_at=np.cumsum(sorted_retrievals,axis=1).astype("float") / np.cumsum(np.ones_like(sorted_retrievals),axis=1)
    #R_at = np.cumsum(sorted_retrievals, axis=1).astype("float") / np.maximum(relevant_count,1)

    # If there are no relevant items and the user replied none, he gets 100% for the query
    #print R_at.shape
    #P_at = np.concatenate([relevant_count == 0,P_at],axis=1)
    #R_at = np.concatenate([relevant_count == 0, R_at], axis=1)
    # When there are no relevant items, Recall is always 100 %
    #R_at[relevant_count.reshape(-1) == 0,:] = 1
    print R_at.shape
    plt.plot(P_at.mean(axis=0))*100;plt.plot(R_at.mean(axis=0));plt.show()

    P_by_query=P_at[np.arange(nb_queries,dtype="int64"),stop_indexes]
    R_by_query=R_at[np.arange(nb_queries,dtype="int64"),stop_indexes]
    P = P_by_query.mean()
    R = R_by_query.mean()
    Fm = (2*P*R)/(P+R)
    RoC={"Precision":P_at.mean(axis=0),"Recall":R_at.mean(axis=0)}
    return Fm,P,R,RoC