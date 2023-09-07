# MOTCat
<details>
<summary>
  <b>Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction</b>, ICCV 2023.
  <a href="https://arxiv.org/abs/2306.08330" target="blank">[arxiv]</a>
  <br><em>Yingxue Xu, Hao Chen</em></br>
</summary>

```bash
@article{xu2023multimodal,
  title={Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction},
  author={Xu, Yingxue and Chen, Hao},
  journal={arXiv preprint arXiv:2306.08330},
  year={2023}
}
```
</details>

**Summary:** Here is the official implementation of the paper "Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction".

<img src="imgs/overview.png" width="1500px" align="center" />

### Pre-requisites:
```bash
pot==0.8.2 (important!!!)
torch 1.12.0+cu113
scikit-survival 0.17.2
```
We found there may be bugs in POT package after version 0.8.2. So please make sure you use the version 0.8.2 or the previous 0.7.x version. Otherwise, it may not work.
### Prepare your data
#### WSIs
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 pretrained 1024-dim feature for each 256 $\times$ 256 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

DATA_ROOT_DIR is the base directory of cancer type (e.g. the directory to TCGA_BLCA), which should be passed to the model with the argument `--data_root_dir` as shown in [command.md](./scripts/command.md).

#### Genomics
In this work, we directly use the preprocessed genomic data provided by [MCAT](https://github.com/mahmoodlab/MCAT), stored in folder [dataset_csv](./dataset_csv).

## Training-Validation Splits
Splits for each cancer type are found in the `splits/5foldcv ` folder, which are randomly partitioned each dataset using 5-fold cross-validation. Each one contains splits_{k}.csv for k = 1 to 5. To compare with MCAT, we follow the same splits as that of MCAT.

## Running Experiments
To train MOTCat, you can specify the argument in the bash `train_motcat.sh` stored in [scripts](./scripts/) and run the command:
```bash
sh scripts/train_motcat.sh
```
or use the following generic command-line and specify the arguments:
```bash
CUDA_VISIBLE_DEVICES=<DEVICE_ID> python main.py \
--data_root_dir <DATA_ROOT_DIR> \
--split_dir <SPLITS_FOR_CANCER_TYPE> \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg <OT_ENTROPIC_REGULARIZATION> --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig
```
Commands for all experiments of MOTCat can be found in the [command.md](./scripts/command.md) file.

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [MCAT](https://github.com/mahmoodlab/MCAT)
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [POT](https://github.com/PythonOT/POT)

## License & Citation 
If you find our work useful in your research, please consider citing our paper at:
```bash
@article{xu2023multimodal,
  title={Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction},
  author={Xu, Yingxue and Chen, Hao},
  journal={arXiv preprint arXiv:2306.08330},
  year={2023}
}
```
This code is available for non-commercial academic purposes. If you have any question, feel free to email [Yingxue XU](https://innse.github.io/).