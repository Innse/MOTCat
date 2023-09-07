from __future__ import print_function
import argparse
import os
import random
import sys

from timeit import default_timer as timer
import numpy as np

# Internal Imports
from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

# PyTorch Imports
import torch


### Training settings
parser = argparse.ArgumentParser(
    description='Configurations for Survival Analysis on TCGA Data.')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1,
                    help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1,
                    help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results',
                    help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_blca',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
parser.add_argument('--log_data',        action='store_true', 
                    help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--load_model',        action='store_true',
                    default=False, help='whether to load model')
parser.add_argument('--path_load_model', type=str,
                    default='/path/to/load', help='path of ckpt for loading')
parser.add_argument('--start_epoch',              type=int,
                    default=0, help='start_epoch.')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['snn', 'amil', 'mcat', 'motcat'], 
                    default='motcat', help='Type of model (Default: motcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                    default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=[
                    'None', 'concat'], default='concat', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true',
                    default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true',
                    default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str,
                    default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str,
                    default='small', help='Network size of SNN model')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str,
                    choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int,
                    default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20,
                    help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4,
                    help='Learning rate (default: 0.0002)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv',
                    'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: nll_surv)')
parser.add_argument('--label_frac',      type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', 			 type=float, default=1e-5,
                    help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0,
                    help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                    help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true',
                    default=False, help='Enable early stopping')

### MOTCat Parameters
parser.add_argument('--bs_micro',      type=int, default=256,
                    help='The Size of Micro-batch (Default: 256)') ### new
parser.add_argument('--ot_impl', 			 type=str, default='pot-uot-l2',
                    help='impl of ot (default: pot-uot-l2)') ### new
parser.add_argument('--ot_reg', 			 type=float, default=0.1,
                    help='epsilon of OT (default: 0.1)')
parser.add_argument('--ot_tau', 			 type=float, default=0.5,
                    help='tau of UOT (default: 0.5)')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'data_root_dir': args.data_root_dir,
            'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,
            'ot_reg': args.ot_reg,
            'ot_tau': args.ot_tau,
            'ot_impl': args.ot_impl,
            'bs_micro': args.bs_micro}

print('\nLoad Dataset')
if 'survival' in args.task:
    args.n_classes = 4
    combined_study = '_'.join(args.task.split('_')[:2])

    if combined_study in ['tcga_blca', 'tcga_brca','tcga_gbmlgg', 'tcga_ucec', 'tcga_luad']:
        csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study)
    else:
        csv_path = './%s/%s_all_clean.csv' % (args.dataset_path, combined_study)
    dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
                                           mode=args.mode,
                                           apply_sig=args.apply_sig,
                                           data_dir=args.data_root_dir,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           patient_strat=False,
                                           n_bins=4,
                                           label_col='survival_months',
                                           ignore=[])
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
if args.model_type == 'motcat':
    exp_code = str(args.exp_code) + '_s{}'.format(args.seed)
    if 'uot' in args.ot_impl:
        ot_impl = '_UOT'
        exp_code += ot_impl + '_reg{}'.format(args.ot_reg) + '_tau{}'.format(args.ot_tau) + '_microb{}'.format(args.bs_micro)
    elif 'sinkhorn' in args.ot_impl:
        ot_impl = '_SINKHORN'
        exp_code += ot_impl + '_reg{}'.format(args.ot_reg) + '_microb{}'.format(args.bs_micro)
    else:
        raise NotImplementedError
else:
    exp_code = str(args.exp_code) + '_s{}'.format(args.seed)

print("===="*30)
print("Experiment Name:", exp_code)
print("===="*30)

args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)
print("logs saved at ", args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

# Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

def main(args):
    # Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)
    
    # Start 5-Fold CV Evaluation.
    run_folds = folds
    summary_all_folds = {}
    for i in folds:
        start_t = timer()
        seed_torch(args.seed)
        args.results_pkl_path = os.path.join(
            args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(args.results_pkl_path) and not args.load_model and not args.overwrite:
            print("Skipping Split %d" % i)
            aim_index = np.where(run_folds == i)[0][0]
            run_folds = np.delete(run_folds, aim_index)
            continue
        # Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False,
                                                           csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print('training: {}, validation: {}'.format(
            len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        ### Specify the input dimension size if using genomic features.
        if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)
        elif 'coattn' in args.mode:
            args.omic_sizes = train_dataset.omic_sizes
            print('Genomic Dimensions', args.omic_sizes)
        else:
            args.omic_input_dim = 0
        
        # Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            summary_results, print_results = train(datasets, i, args) ### new

        # Write Results for Each Split to PKL
        save_pkl(args.results_pkl_path, summary_results)
        summary_all_folds[i] = print_results
        end_t = timer()
        print('Fold %d Time: %f seconds' % (i, end_t - start_t))

    print('=============================== summary ===============================')
    result_cindex = []
    for i, k in enumerate(summary_all_folds):
        c_index = summary_all_folds[k]['result'][0]
        print("Fold {}, C-Index: {:.4f}".format(k, c_index))
        result_cindex.append(c_index)
    result_cindex = np.array(result_cindex)
    print("Avg C-Index of {} folds: {:.3f}, stdp: {:.3f}, stds: {:.3f}".format(
        len(summary_all_folds), result_cindex.mean(), result_cindex.std(), result_cindex.std(ddof=1)))

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))