import os
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use("seaborn-darkgrid")

import glob
import numpy as np

#cat ./gt/all_gt.csv |awk "-F\t" '{print $1 "," $5}' |tail +2 > ./results/image/gt_image.csv

lines = [l.split("\t") for l in open("gt/all_gt.csv").read().strip().split("\n")[1:]]
sample_ids = {line[0]:int(line[1]) for line in lines}
writer_ids = {line[0]:int(line[2]) for line in lines}
page_ids = {line[0]:int(line[3]) for line in lines}
image_ids = {line[0]:int(line[4]) for line in lines}
surface_ids = {line[0]:int(line[5]) for line in lines}
originalpixels_ids = {line[0]:int(line[6]) for line in lines}
rectangle_ids = {line[0]:int(line[7]) for line in lines}

rectangle_surface = []
rectangle_density = []
rectangle_original = []

nonrectangle_surface = []
nonrectangle_density = []
nonrectangle_original = []

for sid in rectangle_ids.keys():
    if rectangle_ids[sid]:
        rectangle_surface.append(surface_ids[sid])
        rectangle_original.append(originalpixels_ids[sid])
    else:
        nonrectangle_surface.append(surface_ids[sid])
        nonrectangle_original.append(originalpixels_ids[sid])

rectangle_surface=np.array(rectangle_surface,dtype="float")
rectangle_original=np.array(rectangle_original,dtype="float")
nonrectangle_surface=np.array(nonrectangle_surface,dtype="float")
nonrectangle_original=np.array(nonrectangle_original,dtype="float")
print(np.percentile(rectangle_surface,[25,50,75]))
print(np.percentile(nonrectangle_surface,[25,50,75]))

print("\nRectangle density:",(rectangle_original/rectangle_surface).mean())
print("Non-Rectangle density:",(nonrectangle_original/nonrectangle_surface).mean())

sort_keys_by =lambda x:[t[1] for t in sorted([(v,k) for k,v in x.items()])]

methods = glob.glob("results/writer/ap_*.csv")


#   ./results/writer/ap_srslbp.csv
labels_dict={
    "ap_chahi_model1writer.csv": "TwoPath (writer)",
    "ap_chahi_model2page.csv": "TwoPath (page)",
    "ap_chammas.csv": "ResNet20",
    "ap_gattal_correlation.csv":"oBIF",
    "ap_sheng.csv":"FragNet",
    "ap_srslbp.csv": "SRS-LBP (baseline)",
    "ap_sift.csv": "SIFT (baseline)",
}

p1={}
p10={}
p100={}
ap={}
for method_filename in methods:
    #method_name = method_filename[method_filename.rfind("/ap_"):method_filename.rfind("."):]
    method_name = labels_dict[method_filename.split("/")[-1]]
    per_sample = [l.split(",") for l in open(method_filename).read().strip().split("\n")]
    ap[method_name] = {l[0]: float(l[1]) for l in per_sample}
    p1[method_name] = {l[0]: float(l[2]) for l in per_sample}
    p10[method_name] = {l[0]: float(l[3]) for l in per_sample}
    p100[method_name] = {l[0]: float(l[4]) for l in per_sample}

smooth = 500
stride = 30
figsize=(8,6)

rectangle_perf = []
nonrectangle_perf = []
both_perf = []
method_names = []
#algorithm_p1 = {}
#algorithm_p10 = {}
#algorithm_p100 = {}
plt.clf()
plt.figure(figsize=figsize)
plt.grid(True, which="both", ls="-", axis="y")
for method_name in sorted(ap.keys()):
    label=method_name#" ".join([n[0].upper()+n[1:].lower() for n in method_name.replace("/ap_","").split("_")])
    rectangle_ids_list = sort_keys_by(rectangle_ids)
    is_rectangle = np.array([rectangle_ids[sid] for sid in rectangle_ids_list])==1
    performances = np.array([ap[method_name][sid] for sid in rectangle_ids_list]).reshape(-1)
    rectangle_perf.append(100 * performances[is_rectangle].mean())
    nonrectangle_perf.append(100 * performances[~is_rectangle].mean())
    both_perf.append(100 * performances.mean())
    method_names.append(label)
    #algorithm_p1[method_name] = np.array([p1[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    #algorithm_p10[method_name] = np.array([p10[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    #algorithm_p100[method_name] = np.array([p100[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    #performance = np.convolve(originalpixels_ap[method_name], np.ones((smooth,)) / smooth, mode="valid")
    #original_pixels=original_pixels[smooth//2:-smooth//2+1]
    #plt.semilogx(original_pixels[::stride], 100*performance[::stride], label=label)
    #sns.lineplot(original_pixels[::stride], performance[::stride], label=label)
print("Is Rectangle:",is_rectangle.mean())
barWidth = 0.25
r1 = np.arange(len(rectangle_perf))
r2 = r1+barWidth
r3 = r2+barWidth
#plt.bar(r1, rectangle_perf, color='#7f6d5f', width=barWidth, edgecolor='white', label='Rectangle')
#plt.bar(r2, nonrectangle_perf, color='#557f2d', width=barWidth, edgecolor='white', label='Random Shape')
plt.bar(r1, rectangle_perf, color='darkblue', width=barWidth, edgecolor='white', label='Rectangle')
plt.bar(r2, nonrectangle_perf, color='#557f2d', width=barWidth, edgecolor='white', label='Random Shape')
plt.bar(r3, both_perf, color='purple', width=barWidth, edgecolor='white', label='All')

plt.xticks([r + barWidth for r in range(len(method_names))], method_names, rotation=20)
#plt.setp(xtickNames, rotation=45, fontsize=8)
plt.ylabel("mAP %")

#plt.title(f"Method mAP per sample generation type")
plt.legend()
out_filename = "./plots/rectangle.pdf"
plt.savefig(out_filename)
os.system(f"pdfcrop {out_filename} {out_filename}")



originalpixels_ap = {}
originalpixels_p1 = {}
originalpixels_p10 = {}
originalpixels_p100 = {}


plt.clf()
plt.figure(figsize=figsize)
for method_name in sorted(ap.keys()):
    label=method_name#" ".join([n[0].upper()+n[1:].lower() for n in method_name.replace("/ap_","").split("_")])

    originalpixels_ids_list = sort_keys_by(originalpixels_ids)
    original_pixels=np.array([originalpixels_ids[sid] for sid in originalpixels_ids_list])

    originalpixels_ap[method_name] = np.array([ap[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    originalpixels_p1[method_name] = np.array([p1[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    originalpixels_p10[method_name] = np.array([p10[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    originalpixels_p100[method_name] = np.array([p100[method_name][sid] for sid in originalpixels_ids_list]).reshape(-1)
    performance = np.convolve(originalpixels_ap[method_name], np.ones((smooth,)) / smooth, mode="valid")
    original_pixels=original_pixels[smooth//2:-smooth//2+1]
    plt.semilogx(original_pixels[::stride], 100*performance[::stride], label=label)
plt.ylim(bottom=0.0)
plt.grid(True, which="both", ls="-")
    #sns.lineplot(original_pixels[::stride], performance[::stride], label=label)
plt.xlabel("Query Valid Pixel#")
plt.ylabel("AP by query %")
#plt.title(f"AP by query's Original Pixel Count smoothed by {smooth} samples")
plt.legend()
out_filename = "./plots/original_writer_ap.pdf"
plt.savefig(out_filename)
os.system(f"pdfcrop {out_filename} {out_filename}")


surface_ap = {}
surface_p1 = {}
surface_p10 = {}
surface_p100 = {}

plt.clf()
plt.figure(figsize=figsize)
for method_name in sorted(ap.keys()):
    label=method_name#" ".join([n[0].upper()+n[1:].lower() for n in method_name.replace("/ap_","").split("_")])
    surface_ids_list = sort_keys_by(surface_ids)
    surface=np.array([surface_ids[sid] for sid in surface_ids_list])

    surface_ap[method_name] = np.array([ap[method_name][sid] for sid in surface_ids_list]).reshape(-1)
    surface_p1[method_name] = np.array([p1[method_name][sid] for sid in surface_ids_list]).reshape(-1)
    surface_p10[method_name] = np.array([p10[method_name][sid] for sid in surface_ids_list]).reshape(-1)
    surface_p100[method_name] = np.array([p100[method_name][sid] for sid in surface_ids_list]).reshape(-1)
    performance = np.convolve(surface_ap[method_name], np.ones((smooth,)) / smooth, mode="valid")
    surface = surface[smooth//2:-smooth//2+1]
    plt.semilogx(surface[::stride], 100*performance[::stride], label=label)
#deltasurface=surface[smooth:]-surface[:-smooth]
#deltasurface=200*deltasurface/deltasurface.max()
#plt.semilogy(surface[smooth//2:-smooth//2],deltasurface)
plt.ylim(bottom=0.0)
plt.grid(True, which="both", ls="-")

    #sns.lineplot(original_pixels[::stride], performance[::stride], label=label)
plt.xlabel("Query Surface in Pixels")
plt.ylabel("AP by query %")
#plt.title(f"AP by query's Surface smoothed by {smooth} samples")
plt.legend()
out_filename = "./plots/surface_writer_ap.pdf"
plt.savefig(out_filename)
os.system(f"pdfcrop {out_filename} {out_filename}")
