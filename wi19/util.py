import json
import os
import glob
import numpy as np
import sys
import re
from collections import defaultdict


def get_arg_switches(default_switches, argv=None):
    """Parses arguments for all non-positional parameters.

    :param default_switches: A dictionary defining switches, their default values and optionally help strings.
    :param argv: A list of strings assumed to be like sys.argv. All switches (elements starting whith a hyphen)
        are removed.
    :return: The default_switches updated by argv.
    """
    new_default_switches={}
    switches_help = {"help":"Print help and exit."}
    for k,v in default_switches.items():
        if  hasattr(v, '__len__') and len(v)==2 and isinstance(v[1], basestring):
            switches_help[k]=v[1]
            new_default_switches[k]=v[0]
        else:
            switches_help[k] = ""
            new_default_switches[k] = v
    default_switches=new_default_switches
    del new_default_switches

    default_switches = dict(default_switches, **{"help": False})
    if argv is None:
        argv = sys.argv
    argv_switches = dict(default_switches)
    argv_switches.update([(arg[1:].split("=") if "=" in arg else [arg[1:], "True"]) for arg in argv if arg[0] == "-"])
    if set(argv_switches.keys()) > set(default_switches.keys()):
        help_str = "\n" + argv[0] + " Syntax:\n\n"
        for k in default_switches.keys():
            help_str += "\t-%s=%s %s Default %s.\n" % (
                k, repr(type(default_switches[k])), switches_help[k], repr(default_switches[k]))
        help_str += "\n\nUrecognized switches: "+repr(tuple( set(default_switches.keys()) - set(argv_switches.keys())))
        help_str += "\nAborting.\n"
        sys.stderr.write(help_str)
        sys.exit(1)

    # Setting argv element to the value type of the default.
    argv_switches.update({k: type(default_switches[k])(argv_switches[k]) for k in argv_switches.keys() if type(default_switches[k]) != str and type(argv_switches[k]) == str})

    positionals = [arg for arg in argv if arg[0] != "-"]
    argv[:] = positionals

    help_str = "\n" + argv[0] + " Syntax:\n\n"

    for k in default_switches.keys():
        help_str += "\t-%s=%s %s Default %s . Passed %s\n" % (
        k, repr(type(default_switches[k])), switches_help[k], repr(default_switches[k]), repr(argv_switches[k]))
    help_str += "\nAborting.\n"

    #replace {blabla} with argv_switches["balbla"] values
    replacable_values=["{"+k+"}" for k in argv_switches.keys()]
    while len(re.findall("{[a-z0-9A-Z_]+}","".join([v for v in argv_switches.values() if isinstance(v,str)]))):
        for k,v in argv_switches.items():
            if isinstance(v,str):
                argv_switches[k]=v.format(**argv_switches)

    if argv_switches["help"]:
        sys.stderr.write(help_str)
        sys.exit()
    del argv_switches["help"]

    return argv_switches, help_str


def abort(msg):
    sys.stderr.write("\n{}\nAborting ...\n".format(msg))
    sys.exit(1)


def validate_matrix_is_distance(M):
    """Validates whether M is a valid distance or similarity matrix.

    :param M: a numpy matrix which is either as distance-matrix or a similarity matrix. If all elements in the diagonal
        are the greatest ones row and column wise, the matrix is assumed to be a similarity matrix and a distance id
        they are the smallest values. If neither applies a ValueError is raised.
    :return:
    """
    assert M.shape[0]==M.shape[1] and len(M.shape)==2 and M.shape[0]>2
    assert np.all(M == M.T)

    # In order to tolerate zero distance outside the diagonal, this e is added.
    e = .00000000000001
    sz = M.shape[0]

    min_horiz = np.argmin(M - np.eye(sz) * e, axis=0)
    min_vert = np.argmin(M - np.eye(sz) * e, axis=1)

    is_min = np.all((min_horiz==min_vert) & (min_horiz==np.arange(M.shape[0])))
    if is_min:
        return True # The matrix is a distance matrix.

    max_horiz = np.argmax(M + np.eye(sz) * e, axis=0)
    max_vert = np.argmax(M + np.eye(sz) * e, axis=1)
    is_max=np.all((max_horiz==max_vert) & (max_horiz==np.arange(M.shape[0])))
    if is_max:
        return False # The matrix is a similarity matrix.

    raise ValueError("M is neither a distance or similarity matrix")


def load_dm(dm_fname,gt_fname,allow_similarity=True,allow_missing_samples=False,allow_non_existing_samples=False):
    """Loads a distance matrix and a ground-truth tsv.


    :param dm_fname: The file containing the distance matrix of samples of a dataset for evaluation. The file can be
        either in JSON, CSV, or TSV formats and must have a size of N rows \times N+1 columns. The first column must be
        the name of each sample. The remaining columns are float numbers with the distance of the sample in the first
        column and every other sample. The columns represent samples arranged in the same order as the rows.
    :param gt_fname: The path to the csv file containing the ground-truth specificaly the first column has a filename
        and the second column the class number.
    :param allow_similarity: If True, the matrix will be interpreted as a similarity and will be negated.
    :param allow_missing_samples: If True the samples can be omitted from dm_fname.
    :param allow_non_existing_samples: If True the samples can have ids that are not mentioned in gt_fname.
    :return: a tuple containing the distance matrix, a vector with the sample_identities for every row (and column) in
        the distance matrix, and the a vector with the class id of every sample.
    """
    print "Loading submission {} with groundtruth {} ... ".format(dm_fname, gt_fname),
    fname2sample=lambda x: os.path.basename(x.strip()).split(".")[0]

    id_class_tuples=[l.split(",") for l in open(gt_fname).read().strip().split("\n")]
    id2class_dict = {fname2sample(k):int(v) for k,v in id_class_tuples}

    sample_per_class=defaultdict(lambda:[])
    for k,v in id2class_dict.items():
        sample_per_class[v].append(k)
    max_item_per_class_count = max([len(v) for v in sample_per_class.values()])

    if dm_fname.lower().endswith(".json"):
        str_table=json.load(open(dm_fname))
    elif dm_fname.lower().endswith(".csv"):
        lines=open(dm_fname).read().strip().split("\n")
        str_table=[l.strip().split(",") for l in lines]
    elif dm_fname.lower().endswith(".tsv"):
        lines=open(dm_fname).read().strip().split("\n")
        str_table=[l.strip().split("\t") for l in lines]
    else:
        raise ValueError("Unknown file format '.{}' for {}.".format(dm_fname.split(".")[-1],dm_fname))

    try:
        if any([len(line)!=2+len(str_table) for line in str_table]):
            raise ValueError() # we dont have 2 more columns labels+valid retrivals.
        relevance_estimate = np.array([int(line[1]) for line in str_table],dtype="int64")
        #removing the relevant_estimate column if successfully parced
        str_table = [line[:1]+line[2:] for line in str_table]
    except ValueError:
        nb_samples=len(str_table)
        relevance_estimate=np.array([max_item_per_class_count]*nb_samples,dtype="int64")

    sample_ids=np.array([fname2sample(line[0]) for line in str_table])
    numerical_table=[[float(col) for col in line[1:]] for line in str_table]
    numerical_table=np.array(numerical_table,dtype="double")
    try:
        is_distance=validate_matrix_is_distance(numerical_table)
        if is_distance:
            dm = numerical_table
        else:
            if allow_similarity:
                dm = -numerical_table
            else:
                abort("Distance matrix given was in fact a similarity matrix which is not allowed.")
    except ValueError:
        abort("Distance matrix is incorrect the diagonal should either contain minimal or maximal values and it should be symmetric.")


    if allow_non_existing_samples:
        # Removing samples unknown by the groundtruth both row and column wise
        keep_idx=np.array([True if sample in id2class_dict.keys() else False for sample in sample_ids])
        sample_ids=sample_ids[keep_idx]
        dm = dm[:,keep_idx][keep_idx,:]
        relevance_estimate = relevance_estimate[keep_idx]
    else:
        assert set(sample_ids)==set(id2class_dict.keys())

    if allow_missing_samples:
        if set(sample_ids)!=set(id2class_dict.keys()):
            abort("{} should contain a row and column for each sample in {}.".format(dm_fname,gt_fname))
    else: # allow_missing_samples
        new_relevance_estimate = np.ones(len(id2class_dict)) * max_item_per_class_count
        new_relevance_estimate[:len(sample_ids)] = relevance_estimate
        relevance_estimate = new_relevance_estimate
        missing=sorted(set(id2class_dict.keys())-set(sample_ids))
        new_dm=np.ones([len(id2class_dict),len(id2class_dict)])*dm.min()
        new_dm[:len(sample_ids),:len(sample_ids)]=dm
        new_sample_ids=np.concatenate([sample_ids,np.array(missing)],axis=0)
        dm = new_dm
        sample_ids = new_sample_ids

    classes = np.array([id2class_dict[id] for id in sample_ids],dtype="int64")
    print " done."
    return dm, relevance_estimate.astype("int64"), sample_ids, classes