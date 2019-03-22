## ICDAR 2019 Writer Identification Competition:

### Install:
```bash
pip install --user --upgrade git+https://github.com/anguelos/dagtasets
```

### Evaluate: 
 
### Leaderboard:
[curent leaderboard](https://anguelos.github.io/wi19_evaluate/)


```bash
GITROOT="/home/anguelos/work/src/wi19_evaluate/"
OUTPUT_ROOT="${GITROOT}/docs/" 
USER_DIRS='/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/team*svg'
GROUNDTRUTH='/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/gt.csv'
INCOMING_ROOT='/home/anguelos/work/src/wi19_evaluate/test_data/test_leaderboard/incoming/'

for FILE_IN in "${INCOMING_ROOT}"team_*/*.csv; 
do 
    FILE_MOVE="$(echo $FILE_IN| sed 's/incoming//g' )"
    FILE_MOVE="${FILE_MOVE%.*}_$(date +"%YmdHM").csv"
    echo "mv ${FILE_IN} ${FILE_MOVE}"; 
done

./bin/wi19leaderboard \
    "-output_root=${OUTPUT_ROOT}" \
    "-user_dirs=${OUTPUT_ROOT}" 
 
git "--git-dir=${GITROOT}.git" add "${OUTPUT_ROOT}/index.html" \
    "${OUTPUT_ROOT}"/svg/*.svg
git "--git-dir=${GITROOT}.git" commit -m "Auto Update"
git "--git-dir=${GITROOT}.git" push
```
