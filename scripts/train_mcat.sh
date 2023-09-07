CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/feature \
--split_dir tcga_brca \
--model_type mcat \
--which_splits 5foldcv \
--apply_sig