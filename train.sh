## CASIA-B
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port=29501 opengait/main.py --cfgs ./configs/HTR_LTR/CASIAB.yaml --phase train

## CCPG
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port=29501 opengait/main.py --cfgs ./configs/HTR_LTR/CCPG.yaml --phase train

## Gait3D
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port=29501 opengait/main.py --cfgs ./configs/HTR_LTR/Gait3D.yaml --phase train

## OULP
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port=29501 opengait/main.py --cfgs ./configs/HTR_LTR/OULP.yaml --phase train