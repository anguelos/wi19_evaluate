This repository has the bilde submodule.
Clone with recurcive submodules 

```bash
git clone --recurse-submodules .....
```

This code implements the paper 'Sparse Radial Sampling LBP for Writer Identification'

https://arxiv.org/pdf/1504.06133.pdf

Bibtex:
```bash
@inproceedings{nicolaou2015sparse,
  title={Sparse radial sampling LBP for writer identification},
  author={Nicolaou, Anguelos and Bagdanov, Andrew D and Liwicki, Marcus and Karatzas, Dimosthenis},
  booktitle={2015 13th International Conference on Document Analysis and Recognition (ICDAR)},
  pages={716--720},
  year={2015},
  organization={IEEE}
}
```


Build the feature extractor:
```bash
cd bilde/src
make ./lbpFeatures2
```

Extract the features:
```bash
./bin/src/lbpFeatures2 -T otsu -r 1 2 3 4 5 6 7 8 9 10 11 12 -s bilinear -i ./wi_comp_19_validation/*.jpg > /tmp/features.csv
```

Produce the submission:
```bash
./srslbp/srs_lbp.py -validation_csv=/tmp/features.csv -output=/tmp/submission.csv
```

Making a sharded makefile for parallel execution:
```python
shard_csv=lambda x:"./features/piece{}.csv".format(x)
def split_ds(filenames,shards):
    res_makefile="all: "+" ".join([shard_csv(n) for n in range(shards)])+"\n\n"
    shard_sz=int(len(filenames)/(shards))+1
    for n,shard_start in enumerate(range(0,len(filenames),shard_sz)):
        res_makefile+="{0}:\n\t{2} -T otsu -r 1 2 3 4 5 6 7 8 9 10 11 12 -s bilinear -i {1} > {0}\n\n".format(shard_csv(n)," ".join(filenames[shard_start:shard_start+shard_sz]),"./wi19_evaluate/srslbp/bilde/src/lbpFeatures2")
    return res_makefile
open("Makefile","w").write(split_ds(filenames,1000))
```
