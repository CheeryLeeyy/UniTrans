# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def check_missing_key(model_state_dict, ckpt_state_dict):
    checkpoint_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    missing_key_modules = set([keyname.split('.')[0] for keyname in missing_keys])
    extra_key_modules = set([keyname.split('.')[0] for keyname in extra_keys])

    print("------ Loading Checkpoint ------")
    if len(missing_key_modules) == 0 and len(extra_key_modules) ==0:
        return

    print("Missing keys from ckpt:")
    print(*missing_key_modules,sep='\n',end='\n\n')
    # print(*missing_keys,sep='\n',end='\n\n')

    print("Extra keys from ckpt:")
    print(*extra_key_modules,sep='\n',end='\n\n')
    print(*extra_keys,sep='\n',end='\n\n')

    print("You can go to tools/train_utils.py to print the full missing key name!")
    print("--------------------------------")


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_
    
    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
        # check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        # check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model



def load_saved_model_safe(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    def load_state_dict_strict(model, loaded_state_dict):
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for name, param in loaded_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    new_state_dict[name] = param  # Shape matches, load the weight
                else:
                    print(
                        f"Skipping parameter '{name}' as the loaded weight shape {param.shape} does not match the expected shape {model_state_dict[name].shape}.")
            else:
                print(f"Skipping parameter '{name}' as no matching layer was found in the model.")

        # Load the new state_dict with strict=False
        model.load_state_dict(new_state_dict, strict=False)

    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("resuming best validation model at epoch %d" % \
                  eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
            loaded_state_dict = torch.load(file_list[0], map_location='cpu')
            check_missing_key(model.state_dict(), loaded_state_dict)
            # model.load_state_dict(loaded_state_dict, strict=False)
            # 使用自定义的加载函数
            load_state_dict_strict(model, loaded_state_dict)
            return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = epoch

    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        try:
            # model.load_state_dict(loaded_state_dict, strict=False)
            # 使用自定义的加载函数
            load_state_dict_strict(model, loaded_state_dict)
        except:
            print("")


    return initial_epoch, model

def load_original_model(saved_path, model, protocol=True):
    """
    Load protocol and ego model

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    
    
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    ego_path = os.path.join(saved_path, 'ego.pth')
    
    if protocol:
        protocol_path = os.path.join(saved_path, 'protocol.pth')
        assert os.path.exists(protocol_path), '{} not found'.format(protocol_path)
        print("load protocal checkpoint from %s" % protocol_path)
        loaded_protocol_state_dict = torch.load(protocol_path, map_location='cpu')
        check_missing_key(model.state_dict(), loaded_protocol_state_dict)
        model.load_state_dict(loaded_protocol_state_dict, strict=False)
        print()
    
    assert os.path.exists(ego_path), '{} not found'.format(ego_path)
    print("load ego checkpoint from %s" % ego_path)
    loaded_ego_state_dict = torch.load(ego_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_ego_state_dict)
    model.load_state_dict(loaded_ego_state_dict, strict=False)
    
    return model


def load_3modality_models(saved_path, model, modalities):
    """
    加载多个模态模型参数

    Parameters
    __________
    saved_path : str
       模型保存路径
    model : opencood object
        模型实例
    modalities : list
        要加载的模态列表，如['m2', 'm4', 'm5']

    Returns
    -------
    model : opencood object
        加载多模态参数后的模型
    """
    assert os.path.exists(saved_path), f'{saved_path} not found'

    for modality in modalities:
        # 构建模态路径
        modality_path = os.path.join(saved_path, f'{modality}.pth')

        # 验证路径存在性
        assert os.path.exists(modality_path), \
            f'{modality} checkpoint {modality_path} not found'

        print(f"Loading {modality} checkpoint from {modality_path}")

        # 加载参数
        loaded_state_dict = torch.load(modality_path, map_location='cpu')

        # 检查键匹配
        check_missing_key(model.state_dict(), loaded_state_dict)

        # 加载到模型
        model.load_state_dict(loaded_state_dict, strict=False)

    return model


def load_merged_model(saved_path, model):
    """
    Load protocol and ego model

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """

    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    model_path = os.path.join(saved_path, 'merged.pth')
    assert os.path.exists(model_path), '{} not found'.format(model_path)
    print("load merged checkpoint from %s" % model_path)
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)

    return model


def load_intrinsic_model(saved_path, model):
    """
    Load protocol and ego model

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """

    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    model_path = os.path.join(saved_path, 'intrinsic.pth')
    assert os.path.exists(model_path), '{} not found'.format(model_path)
    print("load intrinsic checkpoint from %s" % model_path)
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)

    return model


def load_pretrained_model(saved_path, model, pth_name=""):

    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    model_path = os.path.join(saved_path, pth_name)
    assert os.path.exists(model_path), '{} not found'.format(model_path)
    print("load pretrained checkpoint from %s" % model_path)
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)

    return model



def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def create_losses_heter(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    criterion_dict = dict()
    for modality_name in hypes['loss'].keys():
        
        loss_func_name = hypes['loss'][modality_name]['core_method']
        loss_func_config = hypes['loss'][modality_name]['args']

        loss_filename = "opencood.loss." + loss_func_name
        loss_lib = importlib.import_module(loss_filename)
        loss_func = None
        target_loss_name = loss_func_name.replace('_', '')

        for name, lfunc in loss_lib.__dict__.items():
            if name.lower() == target_loss_name.lower():
                loss_func = lfunc

        if loss_func is None:
            print('loss function not found in loss folder. Please make sure you '
                'have a python file named %s and has a class '
                'called %s ignoring upper/lower case' % (loss_filename,
                                                        target_loss_name))
            exit(0)

        criterion_dict[modality_name] = loss_func(loss_func_config)
    return criterion_dict

def create_adapter_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss_adapter']['core_method']
    loss_func_config = hypes['loss_adapter']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion

def create_moe_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss_moe']['core_method']
    loss_func_config = hypes['loss_moe']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion

def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, n_iter_per_epoch=1):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)
    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    else:
        sys.exit('Unidentified scheduler')

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
