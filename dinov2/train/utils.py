import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '/home/chaofeng/dinov2_finetune')
import torch
from dinov2.data import DataAugmentationDINO, MaskingGenerator # , collate_data_and_cast
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.utils.utils import CosineScheduler
from functools import partial
import random

# def build_dataloader_raw(cfg):
#     img_size = cfg.crops.global_crops_size
#     patch_size = cfg.student.patch_size
#     n_tokens = (img_size // patch_size) ** 2
#     mask_generator = MaskingGenerator(          # for ibot mask patch loss
#         input_size=(img_size // patch_size, img_size // patch_size),
#         max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
#     )

#     data_transform = DataAugmentationDINO(
#         cfg.crops.global_crops_scale,
#         cfg.crops.local_crops_scale,
#         cfg.crops.local_crops_number,
#         global_crops_size=cfg.crops.global_crops_size,
#         local_crops_size=cfg.crops.local_crops_size,
#     )
#     inputs_dtype = torch.half
#     collate_fn = partial(
#         collate_data_and_cast,
#         mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#         mask_probability=cfg.ibot.mask_sample_probability,
#         n_tokens=n_tokens,
#         mask_generator=mask_generator,
#         dtype=inputs_dtype,
#     )

#     # setup data loader
#     train_default='ImageNet:split=TRAIN:root=/dev/shm/chaofeng/imagenet/root:extra=/dev/shm/chaofeng/imagenet/extra'

#     dataset = make_dataset(
#         dataset_str=train_default,
#         transform=data_transform,
#         #target_transform=lambda _: (),
#     )

#     start_iter = 0
#     cfg.train.batch_size_per_gpu = 2 
#     sampler_type = SamplerType.SHARDED_INFINITE
#     data_loader = make_data_loader(
#         dataset=dataset,
#         batch_size=cfg.train.batch_size_per_gpu,
#         num_workers=cfg.train.num_workers,
#         shuffle=True,
#         seed=start_iter,  # TODO: Fix this -- cfg.train.seed
#         sampler_type=sampler_type,
#         sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )
#     return data_loader

from dinov2.data.samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from enum import Enum
from typing import Optional
from torch.utils.data import Sampler, Dataset, DataLoader
import logging
import os
from PIL import Image
logger = logging.getLogger("dinov2")



def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    # mask patch
    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)     # mask patch的最大总数
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)              

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list, # 对应patch flatten后的被mask的patch inedex
        "masks_weight": masks_weight,           # 每个image mask总数的倒数
        "upperbound": upperbound,               # mask patch 最大数量
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4

class dinodataset(Dataset):
    def __init__(self, image_dir, transform) -> None:
        self.transform = transform
        self.data = self._read_paths(image_dir)

    def _read_paths(self, image_dir):
        files_tol = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                files_tol.append(file_path)
                
        files_tol = [f for f in files_tol if f.lower().endswith(('jpeg', 'jpg', 'png'))]
        return files_tol
    
    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        data = self.transform(img)
        return data

def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None

def build_dataloader(cfg, inputs_dtype):
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(          # for ibot mask patch loss
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    dataset = dinodataset(image_dir=cfg.train.dataset_path, transform=data_transform)

    # build dataloader
    sampler = _make_sampler(dataset=dataset, type=SamplerType.SHARDED_INFINITE, shuffle=True, seed=123, size=-1)
    data_loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=cfg.train.batch_size_per_gpu,
                    num_workers=cfg.train.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=False,
                    collate_fn=collate_fn,
                    )
    return data_loader


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    
if __name__ == '__main__':
    config_path = '/home/chaofeng/dinov2_finetune/dinov2/configs/train.yaml'
    cfg = OmegaConf.load(config_path)
    # train_loader = build_dataloader_raw(cfg)
    # for batch in train_loader:
    #     print(batch.keys())


    # img_size = cfg.crops.global_crops_size
    # patch_size = cfg.student.patch_size
    # n_tokens = (img_size // patch_size) ** 2
    # mask_generator = MaskingGenerator(          # for ibot mask patch loss
    #     input_size=(img_size // patch_size, img_size // patch_size),
    #     max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    # )

    # data_transform = DataAugmentationDINO(
    #     cfg.crops.global_crops_scale,
    #     cfg.crops.local_crops_scale,
    #     cfg.crops.local_crops_number,
    #     global_crops_size=cfg.crops.global_crops_size,
    #     local_crops_size=cfg.crops.local_crops_size,
    # )
    # dinodata = dinodataset(image_dir='/dev/shm/chaofeng/imagenet/root/train', transform=data_transform)
    
    dl = build_dataloader(cfg, inputs_dtype=torch.float16)
    for batch in dl:
        print(batch)
        #print(batch.keys())
        pass
