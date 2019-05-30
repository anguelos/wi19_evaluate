from __future__ import print_function
import matplotlib.pyplot as plt
import time
from commands import getoutput as go
from .util import *

import hashlib
from .metrics import get_all_metrics
import datetime

def clean_svg_path(p):
    return "./svg/{}".format(p.split("/")[-1])

def clean_csv_path(p):
    return "./csv/{}".format(p.split("/")[-1])

def readable_name(name):
    return " ".join([n.capitalize() for n in name.split("_") if n.lower() != "team"])

def calculate_submission(submission_file,gt_fname,allow_similarity=True, allow_missing_samples=True,allow_non_existing_samples=True,roc_svg_path=None):
    print("Before load_dm,allow_missing_samples=",allow_missing_samples)
    D, relevance_estimate, sample_ids, classes = load_dm(submission_file, gt_fname, allow_similarity=allow_similarity, allow_missing_samples=allow_missing_samples,allow_non_existing_samples=allow_non_existing_samples)
    mAP, Fm, P, R, RoC, accuracy, recall_at = get_all_metrics(relevance_estimate, D, classes)
    res = {"date": time.ctime(os.path.getctime(submission_file))}
    res["timestamp"] = os.path.getctime(submission_file)
    res["map"] = mAP
    res["pr"] = P
    res["rec"] = R
    res["fm"] = Fm
    res["acc"] = accuracy
    #res[roc_svg_path]=
    if roc_svg_path is not None:
        csv_path=roc_svg_path.replace("svg","csv")
        plt.clf()
        plt.plot(RoC["fallout"]*100,RoC["recall"]*100)
        plt.title("RoC")
        plt.ylabel("Recall (TPR) %")
        plt.xlabel("False Positive Rate %")
        plt.gcf().patch.set_alpha(0.0)
        plt.savefig(roc_svg_path)
        res["roc_svg"]=clean_svg_path(roc_svg_path)
        sys.stderr.write("Saving csv:"+csv_path+"\n")
        open(csv_path).write(",".join([str(r) for r in recall_at]))
        res["recall_csv"]=clean_csv_path(csv_path)
    else:
        res["roc_svg"] = "N/A"
        sys.stderr.write("Not saving csv:" + csv_path + "\n")
    return res

def calculate_submissions(submission_file_list,gt_fname,name=None,description_file=None,allow_similarity=True, allow_missing_samples=False,allow_non_existing_samples=False,svg_dir_path=None):
    if name is None:
        name=submission_file_list[0].split("/")[-2]
    time_progress_svg_path = roc_svg_path="{}/{}_progress.svg".format(svg_dir_path,name)
    if description_file is None:
        description="NA"
    else:
        description=open(description_file).read()

    submission_list=[]
    for submission_file in submission_file_list:
        roc_svg_path="{}/{}_{}_roc.svg".format(svg_dir_path,name,hashlib.md5(submission_file).hexdigest())
        submission = calculate_submission(submission_file=submission_file, gt_fname=gt_fname,
                                          allow_similarity=allow_similarity,
                                          allow_missing_samples=allow_missing_samples,
                                          allow_non_existing_samples=allow_non_existing_samples,
                                          roc_svg_path=roc_svg_path)
        submission_list.append(submission)
    reversed_submissions = sorted(submission_list,key=lambda x: x["timestamp"], reverse=True)
    np_map=np.array([s["map"] for s in reversed_submissions])
    np_pr = np.array([s["pr"] for s in reversed_submissions])
    np_rec = np.array([s["rec"] for s in reversed_submissions])
    np_fm = np.array([s["fm"] for s in reversed_submissions])
    np_dates=np.array([s["date"] for s in reversed_submissions])
    np_timestamps = np.array([s["timestamp"] for s in reversed_submissions])
    fig, ax = plt.subplots()
    plt.plot(np_pr[::-1],label="Fallout")
    plt.plot(np_rec[::-1],label="Recall")
    plt.plot(np_fm[::-1],label= "F-Score")
    plt.plot(np_map[::-1],label= "mAP")
    ax.set_xticks(np_timestamps[::-1])
    ax.set_xticklabels(np_dates[::-1])
    plt.title("{} over time".format(name))
    plt.ylabel("Performance %")
    plt.xlabel("Time")
    plt.savefig(time_progress_svg_path)
    res={"submissions":reversed_submissions,"name":name,"best_map":np_map.max()}
    return res


def calculate_participants(participant_dir_list,gt_fname,out_dir):
    print("Participants",participant_dir_list)
    initial_time=time.time()
    svg_dir=out_dir+"./svg/"
    go("mkdir -p "+svg_dir) # TODO (anguelos) remove svg_dir
    participants=[]
    best_maps=[]
    last_maps = []
    names=[]
    for participant_dir in participant_dir_list:
        name = [p for p in participant_dir.split("/") if len(p)][-1]
        filenames=glob.glob(participant_dir+"/*tsv")+glob.glob(participant_dir+"/*csv")#+glob.glob(participant_dir+"/*json")
        description_path=(glob.glob(participant_dir+"/*README*")+glob.glob(participant_dir+"/description"))
        if len(description_path)==1:
            description_path=description_path[0]
        else:
            description_path = None
        report=calculate_submissions(filenames, gt_fname,name=name,description_file=description_path,svg_dir_path=svg_dir)
        maps=[s["map"] for s in report["submissions"]]
        if maps==[]:
            maps=["N/A"]
            best_maps.append("N/A")
            last_maps.append("N/A")
        else:
            best_maps.append(max(maps))
            last_maps.append(maps[0])
        participant={"submissions":report["submissions"],"name":readable_name(name),"best_map":max(maps),"description":open(description_path).read()}
        names.append(readable_name(name))
        participants.append(participant)

    index=np.arange(len(names))
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    fig.patch.set_alpha(0.0)
    bar_width = 0.35
    _ = ax.bar(index, best_maps, bar_width, color='b',label='Best mAP')
    _ = ax.bar(index+bar_width, last_maps, bar_width, color='g', label='Current mAP')
    ax.set_xticks(index + bar_width / 2)
    plt.xticks(rotation=30)
    ax.set_xticklabels(names)
    ax.set_title("ICDAR 2019 Writer Identification Leaderboard")
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.legend()
    participants_svg="{}{}".format(svg_dir, "participants.svg")
    fig.savefig(participants_svg)
    return {"date":datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),"duration":time.time()-initial_time,"names":names,"best_maps":best_maps,"last_maps":last_maps,"participants_svg":clean_svg_path(participants_svg),"participants":participants}


def print_single_submission_report(submission_file,gt_fname,allow_similarity=True, allow_missing_samples=True,allow_non_existing_samples=True,roc_svg_path=""):
    if roc_svg_path == "":
        roc_svg_path = None
    submission = calculate_submission(submission_file=submission_file,gt_fname=gt_fname,allow_similarity=allow_similarity, allow_missing_samples=allow_missing_samples,allow_non_existing_samples=allow_non_existing_samples,roc_svg_path=roc_svg_path)
    print("Submission created on {}".format(submission["date"]))
    print("Preview RoC in bash:\nfirefox {}\n".format(submission["roc_svg"]))
    print("Precision: {:5.3} %\nRecall: {:5.3} %\nF-Score mAP: {:5.3} %\nmAP: {:5.3} %\nAcc.: {:5.3}".format(submission["pr"],submission["rec"],submission["fm"],submission["map"],submission["acc"]))

def print_single_submission_table(submission_file,gt_fname,allow_similarity=True, allow_missing_samples=True,allow_non_existing_samples=True,roc_svg_path=""):
    if roc_svg_path == "":
        roc_svg_path = None
    submission = calculate_submission(submission_file=submission_file,gt_fname=gt_fname,allow_similarity=allow_similarity, allow_missing_samples=allow_missing_samples,allow_non_existing_samples=allow_non_existing_samples,roc_svg_path=roc_svg_path)
    fname2name=lambda x:" ".join(x.split("/")[-1].split(".")[0].split("_"))
    print("Submission & partition & Precision & Recall & F-Score & mAP & Accuracy \\\\ \n {} & {} & {:5.3} \% & {:5.3} \% & {:5.3} \% & {:5.3} \% & {:5.3} \%".format(
        fname2name(submission_file),fname2name(gt_fname),100*submission["pr"],100*submission["rec"],100*submission["fm"],100*submission["map"],100*submission["acc"]))
