# LVSA
Source code of the paper "From Grounding to Skolemization: A Logic-Constrained Vector Symbolic Architecture for Complex Query Answering".

Our code is based on an implementation of CQD available [here](https://github.com/pminervini/KGReasoning/).



## 1. Prepare the datasets

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).



## 2. Prepare the pre-trained embeddings

As analyzed in **Appendix B**, our LVSA aligns with ComplEx in the designs of the embedding space and the triple scoring metric. Therefore, we initialize entity and relation embeddings in LVSA using pretrained ComplEx embeddings. These embeddings are pretrained using the [ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction) codebase.

The pretrained embeddings used in our experiments are available in the supplementary materials (under the `/kge` directory).



## 3. Train neural modules

You can train the differential Skolemization module and the neural negator using the following commands. The trained models will be saved in the `../LVSA_model/` directory:

```shell
bash scripts/train_fb15k.sh

bash scripts/train_fb15k237.sh

bash scripts/train_nell.sh
```



## 4. Evaluation

You can perform complex query answering using LVSA by running the following scripts:

```shell
bash scripts/eval_fb15k.sh

bash scripts/eval_fb15k237.sh

bash scripts/eval_nell.sh
```

