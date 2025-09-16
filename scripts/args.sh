DATASET_NAME=$1

if [[ $DATASET_NAME == "e2h" ]]; then
    DATA_DIR=assets/datasets/edges2handbags
    DATASET=edges2handbags
    IMG_SIZE=64

    NUM_CH=192
    NUM_RES_BLOCKS=3
    ATTN_TYPE=True

    EXP="e2h${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False
    UNET_TYPE="adm"

    PRED="vp"
elif [[ $DATASET_NAME == "diode" ]]; then
    DATA_DIR=assets/datasets/DIODE-256
    DATASET=diode
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True

    EXP="diode${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=16
    DROPOUT=0.1
    CLASS_COND=False

    UNET_TYPE="adm"

    PRED="vp"
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    DATA_DIR=assets/datasets/ImageNet
    DATASET=imagenet_inpaint_center
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=False

    EXP="imagenet_inpaint_center${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=16
    DROPOUT=0
    CLASS_COND=True
    UNET_TYPE='adm'

    PRED="i2sb_cond"

#TODO: Unet
# elif [[ $DATASET_NAME == "sen12mscr" ]]; then
#     DATA_DIR=/home/work/dataset/SEN12MSCR
#     DATASET=sen12mscr
#     IMG_SIZE=256

#     NUM_CH=64
#     NUM_RES_BLOCKS=2
#     ATTN_TYPE=True

#     EXP="sen12mscr${IMG_SIZE}_${NUM_CH}d"
#     SAVE_ITER=20000
#     MICRO_BS=16
#     DROPOUT=0.1
#     CLASS_COND=False
#     UNET_TYPE="adm"
#     PRED="vp"

#TODO: NAFNet, NUM_RES_BLOCKS=NUM_NAF_BLOCKS
elif [[ $DATASET_NAME == "sen12mscr" ]]; then
    DATA_DIR=/home/work/dataset/SEN12MSCR
    DATASET=sen12mscr
    IMG_SIZE=256

    NUM_CH=16
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True

    EXP="sen12mscr${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=4
    DROPOUT=0
    CLASS_COND=False
    UNET_TYPE="naf"
    PRED="vp"

fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
    SIGMA_MAX=80.0
    SIGMA_MIN=0.002
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "i2sb_cond" ]]; then
    EXP+="_i2sb_cond"
    COND=concat
    BETA_MAX=1.0
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi