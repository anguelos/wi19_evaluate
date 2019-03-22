from __future__ import print_function
import jinja2
import matplotlib.pyplot as plt
import time
import os
import glob
from commands import getoutput as go
from .util import *
from .leader_board import leaderboard_template
from .metrics import get_all_metrics

def readable_name(name):
    return " ".join([n.capitalize() for n in name.split("_") if n.lower() != "team"])

def calculate_submission(submission_file,gt_fname,allow_similarity=True, allow_missing_samples=False,allow_non_existing_samples=False,roc_svg_path=None):
    D, relevance_estimate, sample_ids, classes = load_dm(submission_file, gt_fname, allow_similarity=allow_similarity, allow_missing_samples=allow_missing_samples,allow_non_existing_samples=allow_non_existing_samples)
    #mAP = get_map(D,classes)
    #def get_all_metrics(relevant_estimate,D,query_classes,remove_self_column=True, db_classes=None):
    mAP, Fm, P, R, RoC = get_all_metrics(relevance_estimate, D, classes)
    res = {"date": time.ctime(os.path.getctime(submission_file))}
    res["timestamp"] = os.path.getctime(submission_file)
    res["map"] = mAP
    res["pr"] = P
    res["rec"] = R
    res["fm"] = Fm
    if roc_svg_path is not None:
        plt.clf()
        plt.plot(RoC["fallout"]*100,RoC["recall"]*100)
        plt.title("RoC")
        plt.ylabel("Recall (TPR) %")
        plt.xlabel("False Positive Rate %")
        plt.savefig(roc_svg_path)
        res["roc_svg"]=roc_svg_path
    else:
        res["roc_svg"] = "N/A"
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
        roc_svg_path="{}/{}_{}_roc.svg".format(svg_dir_path,name,np.random.randint(1000000,9999999))
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
    res={"submissions":reversed_submissions,"name":name,"description":description,"best_map":np_map.max()}
    return res


def calculate_participants(participant_dir_list,gt_fname,out_dir):
    svg_dir=out_dir+"/svg/"
    go("mkdir -p "+svg_dir) # TODO (anguelos) remove svg_dir
    participants=[]
    best_maps=[]
    last_maps = []
    names=[]
    for participant_dir in participant_dir_list:
        name = [p for p in participant_dir.split("/") if len(p)][-1]
        filenames=glob.glob(participant_dir+"/*tsv")+glob.glob(participant_dir+"/*csv")#+glob.glob(participant_dir+"/*json")
        description_path=glob.glob(participant_dir+"/description")
        if len(description_path)==1:
            description_path=description_path[0]
        else:
            description_path = None
        report=calculate_submissions(filenames, gt_fname,name=name,description_file=description_path,svg_dir_path=svg_dir)
        print(repr(report))
        print(participant_dir)
        maps=[s["map"] for s in report["submissions"]]
        if maps==[]:
            maps=["N/A"]
            best_maps.append("N/A")
            last_maps.append("N/A")
        else:
            best_maps.append(max(maps))
            last_maps.append(maps[0])
        participant={"submissions":report["submissions"],"name":readable_name(name),"best_map":max(maps)}
        names.append(readable_name(name))
        participants.append(participant)

    index=np.arange(len(names))
    plt.clf()
    fig, ax = plt.subplots()
    bar_width = 0.35
    _ = ax.bar(index, best_maps, bar_width, color='b',label='Best mAP')
    _ = ax.bar(index+bar_width, last_maps, bar_width, color='g', label='Current mAP')
    ax.set_xticks(index + bar_width / 2)
    plt.xticks(rotation=30)
    ax.set_xticklabels(names)
    ax.set_title("Leaderboard")
    ax.legend()
    participants_svg="{}{}".format(svg_dir, "participants.svg")
    fig.savefig(participants_svg)
    return {"names":names,"best_maps":best_maps,"last_maps":last_maps,"participants_svg":participants_svg,"participants":participants}


def print_single_submission_report(submission_file,gt_fname,allow_similarity=True, allow_missing_samples=False,allow_non_existing_samples=False,roc_svg_path=""):
    if roc_svg_path == "":
        roc_svg_path = None
    submission = calculate_submission(submission_file=submission_file,gt_fname=gt_fname,allow_similarity=allow_similarity, allow_missing_samples=allow_missing_samples,allow_non_existing_samples=allow_non_existing_samples,roc_svg_path=roc_svg_path)
    print("Submission created on {}".format(submission["date"]))
    print("Preview RoC in bash:\nfirefox {}\n".format(submission["roc_svg"]))
    print("Precision: {:5.3} %\nRecall: {:5.3} %\nF-ScoremAP: {:5.3} %\nmAP: {:5.3} %".format(submission["pr"],submission["rec"],submission["fm"],submission["map"]))


if __name__=="__main__":
    template=jinja2.Template(leaderboard_template)
    open("/tmp/index.html","w").write(template.render(participants=participants))