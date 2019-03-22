import unittest
import tempfile
import numpy as np
import wi19


class TestMatrixLoader(unittest.TestCase):
    def test_simple_csv(self):
        gt_csv = "id1,1\nid2,1\nid3,2\nid4,3\nid5,3\nid6,1"

        subm_csv = "id1,3,.0,.1,.2,.3,.4,.2\n"\
                   "id2,3,.1,.0,.1,.2,.3,.4\n"\
                   "id6,3,.2,.1,.0,.7,.9,.5\n"\
                   "id4,2,.3,.2,.7,.0,.8,.6\n"\
                   "id5,2,.4,.3,.9,.8,.0,.7\n"\
                   "id3,0,.2,.4,.5,.6,.7,.0"

        dm = np.array([[.0, .1, .2, .3, .4, .2],
                       [.1, .0, .1, .2, .3, .4],
                       [.2, .1, .0, .7, .9, .5],
                       [.3, .2, .7, .0, .8, .6],
                       [.4, .3, .9, .8, .0, .7],
                       [.2, .4, .5, .6, .7, .0]], dtype="double")
        labels = np.array(["id1", "id2", "id6", "id5", "id3"])
        relevance = np.array([3, 3, 3, 2, 2, 0], dtype="int64")
        sample2class = {
            "id1": 1,
            "id2": 1,
            "id3": 2,
            "id4": 3,
            "id5": 3,
            "id6": 1}

        fsubm = tempfile.NamedTemporaryFile(suffix=".csv")
        fsubm.write(subm_csv)
        fsubm.flush()
        fgt = tempfile.NamedTemporaryFile(suffix=".csv")
        fgt.write(gt_csv)
        fgt.flush()
        # dm_fname,gt_fname,allow_similarity=True,allow_missing_samples=False,allow_non_existing_samples=False

        o_dm, o_relevance, o_sample2class, _ = wi19.load_dm(
            fsubm.name, fgt.name)

        np.testing.assert_array_equal(
            o_dm, dm, err_msg='Distance matrix wrong.', verbose=True)
        np.testing.assert_array_equal(
            o_relevance,
            relevance,
            err_msg='Relevance vector wrong.',
            verbose=True)
        assert sorted(o_sample2class) == sorted(sample2class.keys())

        # Testing trailing new-line
        fsubm.write("\n")
        fsubm.flush()
        o_dm, o_relevance, o_sample2class, _ = wi19.load_dm(
            fsubm.name, fgt.name)
        np.testing.assert_array_equal(
            o_dm, dm, err_msg='Distance matrix wrong.', verbose=True)
        np.testing.assert_array_equal(
            o_relevance,
            relevance,
            err_msg='Relevance vector wrong.',
            verbose=True)


if __name__ == '__main__':
    unittest.main()
