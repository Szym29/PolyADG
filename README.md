# Poly(A)-DG

Example implementation of the paper:

- Zheng Y, Wang H, Zhang Y, Gao X, Xing EP, Xu M (2020) Poly(A)-DG: A deep-learningbased domain generalization method to identify cross-species Poly(A) signal without prior knowledge from target species. PLoS Comput Biol 16(11): e1008297. https://doi.org/10.1371/journal.pcbi.1008297



## Environments:

```
Python 3.8-3.11
Tensorflow V2
```

## Run

Run `PolyADG.py` directly.

```
python PolyADG.py
```


## Datasets :

- Omni Human Poly(A) signal dataset and BL mouse Poly(A) signal datasets come from [DeeReCT-PolyA](https://github.com/likesum/DeeReCT-PolyA). 
- We established a Rat Poly(A) signal dataset which contains 11 Poly(A) signal motifs. This dataset consists of roughly 37,000 DNA sequences and the number of true Poly(A) signal sequences is the same as pseudo-ones.

Details of every steps to show how we establish the Rat Poly(A) signal dataset can be found in the supplementary of Our paper.

----

## Contact:

[Yumin Zheng](mailto:zhengyumin529@gmail.com)
