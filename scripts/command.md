# MOTCat
## BLCA
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/BLCA/x20 \
--split_dir tcga_blca \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.05 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```

## BRCA
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/BRCA/x20 \
--split_dir tcga_brca \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```

## UCEC
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/UCEC/x20 \
--split_dir tcga_ucec \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```

## GBMLGG
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/GBMLGG/x20 \
--split_dir tcga_gbmlgg \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```

## LUAD
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/LUAD/x20 \
--split_dir tcga_luad \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.05 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```

# MCAT
## BLCA
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/BLCA/x20 \
--split_dir tcga_blca \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
```

## BRCA
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/BRCA/x20 \
--split_dir tcga_brca \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
```

## UCEC
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/UCEC/x20 \
--split_dir tcga_ucec \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
```

## GBMLGG
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/GBMLGG/x20 \
--split_dir tcga_gbmlgg \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
```

## LUAD
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/LUAD/x20 \
--split_dir tcga_luad \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig
```