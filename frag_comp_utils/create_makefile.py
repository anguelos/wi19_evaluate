import glob

lines={}
for gt in glob.glob("./results/*/gt_*.csv"):
    for subm in glob.glob("./submisions/*/dm_*.csv"):
        result_file=gt[:gt.rfind('/')]+subm.replace('dm','ap')[subm.rfind('/'):]
        cmd=f"{result_file}:\n\tPYTHONPATH=./wi19_evaluate/ time python2 ./wi19_evaluate/bin/wi19evaluate -gt_csv={gt} -submission_csv={subm}"
        lines[result_file]=cmd
open("Makefile","w").write("all: "+" ".join(lines.keys())+"\n\n"+"\n\n".join(lines.values())+"\n\nclean:\n\trm -f */*/all_results.csv "+" ".join(lines.keys()))

