# LVSA
Source code of the paper "Existentially Consistent Skolemized Reasoning for Complex Query Answering on Knowledge Graphs".




## 1. Prepare the environment

```shell
$ conda create -n lvsa python=3.9
$ conda activate lvsa
$ pip install torch==2.2.2 numpy==1.24.4
```



## 2. Prepare the datasets

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).



## 3. Prepare the pre-trained embeddings

As analyzed in **Appendix B**, our LVSA aligns with ComplEx in the designs of the embedding space and the triple scoring metric. Therefore, we initialize entity and relation embeddings in LVSA using pretrained ComplEx embeddings. These embeddings are pretrained using the [ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction) codebase with the following commands:

```shell
$ python -u main.py --dataset FB15K --score_rel True --model ComplEx --rank 1000 --learning_rate 0.05 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 100 --seed 114514  

$ python -u main.py --dataset FB15K-237 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 500 --seed 114514  

$ python -u main.py --dataset NELL995 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0.1 --max_epochs 500 --seed 114514  
```



## 4. Train neural modules

You can train the differential Skolemization module and the neural negator using the following commands. The trained models will be saved in the `../LVSA_model/` directory:

```shell
$ bash scripts/train_fb15k.sh

$ bash scripts/train_fb15k237.sh

$ bash scripts/train_nell.sh
```



## 5. Evaluation

You can perform complex query answering using LVSA by running the following scripts:

```shell
$ bash scripts/eval_fb15k.sh

$ bash scripts/eval_fb15k237.sh

$ bash scripts/eval_nell.sh
```

## 6. Supplementary Results

We report the MRR (%) results of LVSA and prior Skolemization-based baselines on FB15k, FB15k-237, and NELL995. $A_p$, $A_n$, and $A$ denote the average scores over EPFO queries, queries involving negation, and all queries, respectively. `-` indicates that the model does not support reasoning over the corresponding query types. The Transformer-based method PPMT [1] requires substantially longer training (estimated >48h per dataset). We report its performance after 24h of training. Under the data split used in the original PPMT paper, its fully trained performance on EPFO queries is comparable to that of LMPNN.

[1] Xingrui Zhuo, Shirui Pan, Jiapu Wang, Gongqing Wu, Zan Zhang, Rui Li, Zizhong Wei, and Xindong Wu. 2025. Progressive Prefix-Memory Tuning for Complex Logical Query Answering on Knowledge Graphs. In IJCAI. 3716–3724.

### FB15k

| Model | $A_p$ | $A_n$ | $A$ |
|:---|:---:|:---:|:---:|
| CQD-CO | 45.3 | 4.6 | 30.8 |
| BetaE | 41.6 | 11.8 | 31.0 |
| ConE | 49.8 | 14.8 | 37.3 |
| PPMT | 16.4 | - | - |
| LMPNN | 50.6 | 20.0 | 39.7 |
| CLMPT | 55.1 | 17.9 | 41.8 |
| **LVSA** | **59.9** | **27.7** | **48.4** |

### FB15k-237

| Model | $A_p$ | $A_n$ | $A$ |
|:---|:---:|:---:|:---:|
| CQD-CO | 19.8 | 1.7 | 13.3 |
| BetaE | 20.9 | 5.5 | 15.4 |
| ConE | 23.4 | 5.9 | 17.1 |
| PPMT | 15.5 | - | - |
| LMPNN | 24.1 | 7.8 | 18.3 |
| CLMPT | 25.9 | 7.9 | 19.4 |
| **LVSA** | **28.8** | **10.9** | **22.4** |

### NELL995

| Model | $A_p$ | $A_n$ | $A$ |
|:---|:---:|:---:|:---:|
| CQD-CO | 28.4 | 1.9 | 18.9 |
| BetaE | 24.6 | 5.9 | 17.9 |
| ConE | 27.1 | 6.4 | 19.7 |
| PPMT | 17.1 | - | - |
| LMPNN | 30.7 | 8.0 | 22.6 |
| CLMPT | 31.3 | 7.0 | 22.6 |
| **LVSA** | **32.4** | **9.1** | **24.1** |

## Acknowledgement

Our code is based on an implementation of CQD available [here](https://github.com/pminervini/KGReasoning/).
