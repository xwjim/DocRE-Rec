# DocRE with Reconstruction
This repository is the PyTorch implementation of our DocRE model with reconstruction in AAAI 2021 Paper "Document-Level Relation Extraction with Reconstruction".

# Requirements and Installation
```
python3>=3.6
pytorch>=1.5
scikit-learn>=0.21.2
wandb>=0.95
```

# Dataset 
Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into prepro_data folder.

For the dataset and pretrained embeddings, please download it here, which are officially provided by DocRED: [A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127) and put them into data folder.
# Proprocessing
```
python3 gen_data.py --in_path data --out_path prepro_data
python3 gen_graph_data.py --in_path prepro_data --worker_num 24
```
# Training
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name DynGraph --save_name checkpoint_DynGraph --train_prefix dev_train --test_prefix dev_dev
```
# Testing
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name DynGraph --save_name checkpoint_DynGraph --train_prefix dev_train --test_prefix dev_dev --eval_model True --rel_theta 0.3601
```
# Citation
If you find our work or the code useful, please consider cite our paper using:
```
@inproceedings{docred-rec,
 author = {Wang Xu, Kehai Chen, and Tiejun Zhao},
 booktitle = {The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
 title = {Document-Level Relation Extraction with Reconstruction},
 year = {2021}
}
```
