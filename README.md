## ICDAR 2019 Writer Identification Competition:

### Install:
```bash
pip install --user --upgrade git+https://github.com/anguelos/dagtasets
```

### Evaluate:
```bash
#print help
./bin/wi19evaluate -h
# on run data:
./bin/wi19evaluate -submission_csv=./test_data/dm.json -gt_csv=./test_data/gt.csv
 
``` 
 
### Leaderboard:
[curent leaderboard](https://anguelos.github.io/wi19_evaluate/)


```bash
#!/bin/bash

GITROOT="/home/anguelos/work/src/wi19_evaluate/"
OUTPUT_ROOT="${GITROOT}/docs/"
USER_DIRS="/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/team*"
GROUNDTRUTH='/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/gt.csv'
INCOMING_ROOT='/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/incoming/'

for FILE_IN in "${INCOMING_ROOT}"team_*/*.csv;
do
    FILE_MOVE="$(echo $FILE_IN| sed 's/incoming//g' )"
    FILE_MOVE="${FILE_MOVE%.*}_$(date +'%Y%m%d%H%M').csv"
    echo "mv ${FILE_IN} ${FILE_MOVE}";
done

"${GITROOT}bin/wi19leaderboard" \
       "-output_root=${OUTPUT_ROOT}"  \
       "-user_dirs=${USER_DIRS}"   \
       "-gt_csv=${GROUNDTRUTH}"


cd ${GITROOT}
git  add docs/index.html docs/svg/*.svg
git  commit -m "Auto Update"
git  push
```
