"""
singal trainning script
"""
from omegaconf import OmegaConf
import torch
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
sys.path.insert(0, '/home/chaofeng/dinov2_finetune')

from dinov2.train.model import SSLMetaArch
from utils import (
    build_dataloader,
    build_optimizer,
    build_schedulers,
    apply_optim_scheduler
)

def do_train(cfg, model, train_loader):
    model.train()
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    iteration = 1
    for data in train_loader:
        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses        
        optimizer.zero_grad(set_to_none=True)
        from torch.cuda.amp.autocast_mode import autocast
        with autocast(dtype=torch.float16, enabled=True):
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        
        # loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        if cfg.optim.clip_grad:
            for v in model.student.values():
                torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=cfg.optim.clip_grad)
        optimizer.step()
        model.update_teacher(mom)
        losses_reduced = sum(loss for loss in loss_dict.values())
        print('iteration: ', iteration)
        print('loss: ', losses_reduced)
        print('lr: ', lr)
        iteration = iteration + 1



if __name__ == '__main__':
    config_path = '/home/chaofeng/dinov2_finetune/dinov2/configs/train.yaml'
    cfg = OmegaConf.load(config_path)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    model.float()   
    train_loader = build_dataloader(cfg, inputs_dtype=torch.float32)
    do_train(cfg, model, train_loader)
    """
    nohup /var/lib/anaconda3/envs/det/bin/python /home/chaofeng/dinov2_finetune/dinov2/train/train_singal_new.py > /home/chaofeng/dinov2_finetune/n.log 2>&1 &
    """
    """
    lr schedular
    
    """
    


