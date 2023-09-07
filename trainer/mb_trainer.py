import numpy as np
import torch

import os
from sksurv.metrics import concordance_index_censored
import random
from timeit import default_timer as timer

torch.set_num_threads(2)

def train_loop_survival_coattn_mb(epoch, bs_micro, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=32, args=None):
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        loss = 0.
        all_risk = 0.
        cnt = 0
        
        index_chunk_list = split_chunk_list(data_WSI, bs_micro)
        for tindex in index_chunk_list:
            wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
            hazards, S, Y_hat, A  = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            if args.bag_loss == 'nll_surv':
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            elif args.bag_loss == 'cox_surv':
                loss_micro = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
            else:
                raise NotImplementedError
            loss += loss_micro
            all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
            cnt+=1
            
        loss = loss / cnt
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt # averaged risk
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time
        
        
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 50 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

def validate_survival_coattn_mb(cur, epoch, bs_micro, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()
        slide_id = slide_ids.iloc[batch_idx]
        
        loss = 0.
        all_risk = 0.
        cnt = 0
        with torch.no_grad():
            index_chunk_list = split_chunk_list(data_WSI, bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                hazards, S, Y_hat, A = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
                loss += loss_micro
                all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
                cnt+=1

        loss = loss / cnt
        loss_value = loss.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt # averaged risk
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})
        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    print(val_epoch_str)
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False



def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list