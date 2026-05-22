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

## Acknowledgement

Our code is based on an implementation of CQD available [here](https://github.com/pminervini/KGReasoning/).
