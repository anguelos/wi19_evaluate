import numpy as np
from matplotlib import pyplot as plt


def _get_d_plus_e(D):
    return D.astype("float")
    higheer_precision_type = {
        np.half: np.single,
        np.single: np.double,
        np.double: np.longdouble}
    all_vals = np.sort(
        np.unique(D)).astype(
        higheer_precision_type.get(
            D.dtype,
            np.longdouble))
    return np.min(all_vals[1:] - all_vals[:-1]) / 2


def get_map(D, classes):
    correct_retrievals = classes[None, :] == classes[:, None]
    Dp = D + _get_d_plus_e(D) * correct_retrievals
    sorted_indexes = np.argsort(Dp, axis=1)

    # TODO(anguelos) remove samity check
    assert np.all(sorted_indexes[:, 0] == np.arange(sorted_indexes.shape[0]))
    non_singleton_idx = correct_retrievals.sum(axis=1) > 1

    # removing singletons as queries
    correct_retrievals = correct_retrievals[non_singleton_idx, :]
    sorted_indexes = sorted_indexes[non_singleton_idx, :]

    sorted_indexes = sorted_indexes[:, 1:]  # removing self
    sorted_retrievals = correct_retrievals[np.arange(
        sorted_indexes.shape[0], dtype="int32")[:, None], sorted_indexes]
    sorted_retrievals = sorted_retrievals[:, 1:]
    max_precision = np.cumsum(np.ones_like(sorted_retrievals), axis=1)
    max_precision = np.minimum(
        max_precision,
        sorted_retrievals.sum(
            axis=1)[
            :,
            None])

    max_P_at = np.cumsum(
        sorted_retrievals,
        axis=1).astype("float") / max_precision
    # max_P_at is the # correct retrivals @, divied by # of relevant items.
    AP = (max_P_at * sorted_retrievals) / \
        sorted_retrievals.sum(axis=1)[:, None]
    mAP = AP.mean()
    return mAP


def _get_sorted_retrievals(D, classes, remove_self_column=True, apply_e=False):
    correct_retrievals = classes[None, :] == classes[:, None]
    if apply_e:
        D = D + _get_d_plus_e(D) * correct_retrievals
    sorted_indexes = np.argsort(D, axis=1)
    if remove_self_column:
        sorted_indexes = sorted_indexes[:, 1:]  # removing self
    sorted_retrievals = correct_retrievals[np.arange(
        sorted_indexes.shape[0], dtype="int64")[:, None], sorted_indexes]
    return sorted_retrievals


def _get_precision_recall_matrices(D, classes, remove_self_column=True):
    sorted_retrievals = _get_sorted_retrievals(
        D, classes, remove_self_column=remove_self_column)
    relevant_count = sorted_retrievals.sum(axis=1).reshape(-1, 1)
    precision_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.cumsum(np.ones_like(sorted_retrievals), axis=1)
    recall_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.maximum(relevant_count, 1)
    recall_at[relevant_count.reshape(-1) == 0, :] = 1
    return precision_at, recall_at, sorted_retrievals


def _compute_map(precision_at, sorted_retrievals):
    # Removing singleton queries from mAP computation
    valid_entries = sorted_retrievals.sum(axis=1) > 0
    precision_at = precision_at[valid_entries, :]
    sorted_retrievals = sorted_retrievals[valid_entries, :]
    AP = (precision_at * sorted_retrievals).sum(axis=1) / \
        sorted_retrievals.sum(axis=1)
    return AP.mean()


def _compute_fscore(sorted_retrievals, relevant_estimate):
    relevant_mask = np.cumsum(np.ones_like(
        sorted_retrievals), axis=1) <= relevant_estimate.reshape(-1, 1)
    tp = float((sorted_retrievals * relevant_mask).sum())
    retrieved = relevant_estimate.sum()
    relevant = sorted_retrievals.sum()
    precision = tp / retrieved
    recall = tp / relevant
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall


def _compute_roc(sorted_retrievals):
    # https://en.wikipedia.org/wiki/Receiver_operating_characteristic @ 22/3/2019
    true_positives = sorted_retrievals.sum(axis=0).cumsum().astype("float")
    false_positives = (1-sorted_retrievals).sum(axis=0).cumsum().astype("float")
    relevant = np.ones_like(true_positives) * sorted_retrievals.sum()
    recalls = true_positives / relevant
    fallout = false_positives / (1-sorted_retrievals).sum()# FP+TN
    return {"fallout": np.array(fallout), "recall": np.array(recalls)}


def get_all_metrics(
        relevant_estimate,
        D,
        query_classes,
        remove_self_column=True,
        db_classes=None):
    """Computes all performance metrics.

    :param relevant_estimate: an np.array with an integer estimate of how many retrieved samples are relevant for every
        query.
    :param D: the distance matrix between each query and each sample in the retrieval database.
    :param query_classes: The class labels of each query.
    :param remove_self_column: If the queries are in the retrieval database and must be excluded, this should be True.
    :param db_classes: If None, the columns are assumed to be same class as the rows.
    :return: a tuple with four scalar performance estimates mAP,fscore,precision,recall, and a dictionary containing the
        two array needed for ploting roc.
    """
    assert db_classes is None
    precision_at, recall_at, sorted_retrievals = _get_precision_recall_matrices(
        D, query_classes, remove_self_column=remove_self_column)
    del D
    accuracy = precision_at[:,0].mean()
    mAP = _compute_map(precision_at, sorted_retrievals)
    del precision_at,recall_at,
    fscore, precision, recall = _compute_fscore(
        sorted_retrievals, relevant_estimate)
    roc = _compute_roc(sorted_retrievals)
    return mAP, fscore, precision, recall, roc, accuracy, recall_at.mean(axis=0)
