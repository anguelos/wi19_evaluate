import unittest
import numpy as np
from wi19.metrics import _get_sorted_retrievals, _get_precision_recall_matrices, _compute_map, _compute_fscore

class TestMetrics(unittest.TestCase):
    def test_sorted_retrievals(self):
        in_classes = np.array([0,0,1,1,2,0],dtype="int64")
        in_D=np.array([[.0, .1, .2, .9, .4, .5],
                    [.1, .0, .3, .4, .5, .6],
                    [.2, .3, .0, .5, .6, .7],
                    [.9, .4, .5, .0, .7, .8],
                    [.4, .5, .6, .7, .0, .9],
                    [.5, .6, .7, .8, .9, .0]],dtype="float")
        target_sorted_retrievals = np.array([[1, 1, 0, 0, 1, 0],
                                             [1, 1, 0, 0, 0, 1],
                                             [1, 0, 0, 1, 0, 0],
                                             [1, 0, 1, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0],
                                             [1, 1, 1, 0, 0, 0]],dtype=np.bool)

        sorted_retrievals = _get_sorted_retrievals(in_D, in_classes, remove_self_column=False)
        np.testing.assert_array_equal(sorted_retrievals, target_sorted_retrievals,
                                      err_msg='Sorted Retrievals wrong.', verbose=True)
        # Test remove self column
        sorted_retrievals = _get_sorted_retrievals(in_D, in_classes)
        np.testing.assert_array_equal(sorted_retrievals, target_sorted_retrievals[:,1:],
                                      err_msg='Sorted Retrievals wrong.', verbose=True)

    def test_precision_recall(self):
        in_classes = np.array([0,0,1,1,2,0],dtype="int64")
        in_D=np.array([[.0, .1, .2, .9, .4, .5],
                       [.1, .0, .3, .4, .5, .6],
                       [.2, .3, .0, .5, .6, .7],
                       [.9, .4, .5, .0, .7, .8],
                       [.4, .5, .6, .7, .0, .9],
                       [.5, .6, .7, .8, .9, .0]],dtype="float")


        target_pr = np.array([[1.0 / 1, 2.0 / 2, 2.0 / 3, 2.0 / 4, 3.0 / 5, 3.0 / 6],
                              [1.0 / 1, 2.0 / 2, 2.0 / 3, 2.0 / 4, 2.0 / 5, 3.0 / 6],
                              [1.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4, 2.0 / 5, 2.0 / 6],
                              [1.0 / 1, 1.0 / 2, 2.0 / 3, 2.0 / 4, 2.0 / 5, 2.0 / 6],
                              [1.0 / 1, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6],
                              [1.0 / 1, 2.0 / 2, 3.0 / 3, 3.0 / 4, 3.0 / 5, 3.0 / 6]])

        target_rec = np.array([[1.0 / 3, 2.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3],
                               [1.0 / 3, 2.0 / 3, 2.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3],
                               [1.0 / 2, 1.0 / 2, 1.0 / 2, 2.0 / 2, 2.0 / 2, 2.0 / 2],
                               [1.0 / 2, 1.0 / 2, 2.0 / 2, 2.0 / 2, 2.0 / 2, 2.0 / 2],
                               [1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1],
                               [1.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3, 3.0 / 3, 3.0 / 3]])

        pr, rec = _get_precision_recall_matrices(in_D, in_classes, remove_self_column=False)

        np.testing.assert_array_equal(pr, target_pr,  err_msg='Precision @ wrong.', verbose=True)
        np.testing.assert_array_equal(rec, target_rec, err_msg='Recall @ wrong.', verbose=True)

    def test_compute_map(self):
        in_sorted_retrievals = np.array([[1, 1, 0, 0, 1, 0],
                                         [1, 1, 0, 0, 0, 1],
                                         [1, 0, 0, 1, 0, 0],
                                         [1, 0, 1, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0],
                                         [1, 1, 1, 0, 0, 0]],dtype=np.bool)
        in_pr = np.array([[1.0 / 1, 2.0 / 2, 2.0 / 3, 2.0 / 4, 3.0 / 5, 3.0 / 6],
                          [1.0 / 1, 2.0 / 2, 2.0 / 3, 2.0 / 4, 2.0 / 5, 3.0 / 6],
                          [1.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4, 2.0 / 5, 2.0 / 6],
                          [1.0 / 1, 1.0 / 2, 2.0 / 3, 2.0 / 4, 2.0 / 5, 2.0 / 6],
                          [1.0 / 1, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6],
                          [1.0 / 1, 2.0 / 2, 3.0 / 3, 3.0 / 4, 3.0 / 5, 3.0 / 6]])

        target_map= ((1.0 + 1.0+ 3.0/5) /3 + (1+1+3.0/6) /3 +(1+2.0/4)/2 +(1+2.0/3)/2 + 1+(1+1+1.0)/3)/6

        mAP=_compute_map(in_pr,in_sorted_retrievals)
        np.testing.assert_equal(mAP,target_map)


    def test_compute_fscore(self):
        in_sorted_retrievals = np.array([[1, 0, 0, 1, 0], # 1
                                         [1, 0, 0, 0, 1], # 1
                                         [0, 0, 1, 0, 0], # 1
                                         [0, 1, 0, 0, 0], # 0
                                         [0, 0, 0, 0, 0], # 0
                                         [1, 1, 0, 0, 0]],dtype=np.bool)
        in_relevant_estimate = np.array([1, 2, 3, 1, 0, 2],dtype="int64")
        tp= float(1+1+1+0+0+2)
        relevant = 2+2+1+1+0+2
        retrieved = in_relevant_estimate.sum()
        target_precision = tp/retrieved
        target_recall=tp/relevant
        target_fscore=2*target_precision*target_recall/(target_precision+target_recall)
        fscore,precision,recall=_compute_fscore(in_sorted_retrievals,in_relevant_estimate)
        print fscore,precision,recall
        print target_fscore, target_precision, target_recall
        np.testing.assert_equal(precision,target_precision)
        np.testing.assert_equal(recall, target_recall)
        np.testing.assert_equal(fscore, target_fscore)



if __name__ == '__main__':
    unittest.main()