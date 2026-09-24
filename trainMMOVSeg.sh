export DETECTRON2_DATASETS=$PWD/Dataset
#################### ViT-B/16 #########################################
#### PIEclean  ###
python $PWD/train_net.py --config $PWD/configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 4 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/PIE.json" DATASETS.TEST \(\"PIE_valrgb_sem_seg\"\,\) 

#### PIEcloud ####
python $PWD/train_net.py --config $PWD/configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 4 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/PIE.json" DATASETS.TEST \(\"PIE_val_sem_seg\"\,\) 

#### OEMthin ####
python $PWD/train_net.py --config $PWD/configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/OEM5class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/OEM.json" DATASETS.TEST \(\"OEMthin_val_sem_seg\"\,\) 

#### OEMthick ####
python $PWD/train_net.py --config $PWD/configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/OEM5class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/OEM.json" DATASETS.TEST \(\"OEMthick_val_sem_seg\"\,\)

#### DDSK ####
python $PWD/train_net.py --config $PWD/configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 3 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/DDSK3class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/DDSK.json" DATASETS.TEST \(\"DDSK_val_sem_seg\"\,\) 


#################### ViT-L/14 #########################################

#### PIEclean  ###
python $PWD/train_net.py --config $PWD/configs/vitl_336.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 4 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/PIE.json" DATASETS.TEST \(\"PIE_valrgb_sem_seg\"\,\) 

#### PIEcloud ####
python $PWD/train_net.py --config $PWD/configs/vitl_336.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 4 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/PIE.json" DATASETS.TEST \(\"PIE_val_sem_seg\"\,\) 

#### OEMthin ####
python $PWD/train_net.py --config $PWD/configs/vitl_336.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/OEM5class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/OEM.json" DATASETS.TEST \(\"OEMthin_val_sem_seg\"\,\) 

#### OEMthick ####
python $PWD/train_net.py --config $PWD/configs/vitl_336.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/OEM5class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/OEM.json" DATASETS.TEST \(\"OEMthick_val_sem_seg\"\,\)

#### DDSK ####
python $PWD/train_net.py --config $PWD/configs/vitl_336.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 3 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR $PWD/output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "$PWD/datasets/DDSK3class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$PWD/datasets/DDSK.json" DATASETS.TEST \(\"DDSK_val_sem_seg\"\,\) 
