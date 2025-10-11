import jittor as jt
from jittor.dataset import DataLoader
from datasets.dataset import NPY_datasets
from models.unet import UNet
from models.unet_TTT import UNet_TTT
from models.UNet2Plus import UNet2Plus
from models.UNet3Plus import UNet3Plus

from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
from loader import *

import warnings
warnings.filterwarnings("ignore")
import time
jt.flags.use_cuda = 1

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    jt.flags.use_cuda = 1

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=config.num_workers,
                                drop_last=True)
    
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif config.network == 'unet_ttt':
        model = UNet_TTT(activefunc='gelu',droprate=0,kernel_size=3,n_channels=3, n_classes=1)
    elif config.network == 'unet2plus':
        model = UNet2Plus(n_channels=3, n_classes=1, bilinear=True, is_deconv=True, is_batchnorm=True, is_ds=True)
    elif config.network == 'unet3plus':
        model = UNet3Plus(n_channels=3, n_classes=1, bilinear=True, is_deconv=True, is_batchnorm=True)
    else:
        raise Exception('network in not right!')
    
    # Jittor模型自动在GPU上运行

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model, config.lr)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = jt.load(resume_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        except Exception as e:
            log_info = f'只加载模型权重，优化器加载失败: {str(e)}'
            # 只有模型状态加载成功，其他状态保持初始值

        logger.info(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )


        if loss < min_loss:
            jt.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        jt.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    # 测试最佳模型
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = jt.load(config.work_dir + '/checkpoints/best.pth')
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )   
    

if __name__ == '__main__':
    config = setting_config
    main(config)