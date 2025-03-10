# PolySnake
🔥 **Good news! Our work has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (*TCSVT*), 2024.**

<p>
    <a href='https://arxiv.org/pdf/2301.08898.pdf' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</p>

This is a PyTorch/GPU re-implementation of the paper [Recurrent Generic Contour-based Instance Segmentation with Progressive Learning](https://arxiv.org/pdf/2301.08898.pdf) 
![city](assets/overview.png)

### Results on Instance Segmentation
![image](https://user-images.githubusercontent.com/50725551/223946327-a7316500-b7b6-4842-a89d-f12c9967b117.png)
### Results on Scene Text Detection
![image](https://github.com/fh2019ustc/PolySnake/assets/50725551/265ba8c0-709e-42a9-959e-e7caab0a94fa)
### Results on Lane Detection
![image](https://github.com/fh2019ustc/PolySnake/assets/50725551/cd94ac03-8ba0-4562-9306-c69b04b931c1)


Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md).

## Testing

### Testing on Cityscapes

1. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1RMS9eYafhF4AJV2qZsYhZjekB1Z8jXG7/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1-CbYA-vYbxxXgotuD2Vhww?pwd=7ryd), and put it to `$ROOT/data/model/rcnn_snake/cityscapes/`.
2. Test:
    ```
    # use coco evaluator
    python run.py --type evaluate --cfg_file configs/city_snake.yaml
    # use the cityscapes official evaluator
    python run.py --type evaluate --cfg_file configs/city_snake.yaml test.dataset CityscapesVal
    ```
3. Speed:
    ```
    python run.py --type network --cfg_file configs/city_snake.yaml
    ```

### Testing on Kins

1. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1aqNmy5YFubmvWMBtpjtpdIABkF7LvdBU/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1-CbYA-vYbxxXgotuD2Vhww?pwd=7ryd), and put it to `$ROOT/data/model/snake/kins/`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/kins_snake.yaml test.dataset KinsVal
    ```
3. Speed:
    ```
    python run.py --type network --cfg_file configs/kins_snake.yaml test.dataset KinsVal
    ```

### Testing on Sbd

1. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1LAHF228PNiKYaMUTkIoc7ztzI9SqTjVP/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1-CbYA-vYbxxXgotuD2Vhww?pwd=7ryd), and put it to `$ROOT/data/model/snake/sbd/`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/sbd_snake.yaml test.dataset SbdVal
    ```
3. Speed:
    ```
    python run.py --type network --cfg_file configs/sbd_snake.yaml test.dataset SbdVal
    ```

### Testing on COCO

1. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1SWU3erDgePcIHIOWpW11dJomXb39ORw8/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1-CbYA-vYbxxXgotuD2Vhww?pwd=7ryd), and put it to `$ROOT/data/model/snake/coco/`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/coco_snake.yaml test.dataset CocoVal
    ```
3. Speed:
    ```
    python run.py --type network --cfg_file configs/coco_snake.yaml test.dataset CocoVal
    ```

### Testing on ADE20K

1. Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1OineXWVAmxZvi42qoeEcaX91N2TaOMR7?usp=drive_link), and put it to `$ROOT/data/model/snake/ade20k/`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/ade20k.yaml test.dataset ade20kVal
    ```
3. Speed:
    ```
    python run.py --type network --cfg_file configs/ade20k.yaml test.dataset ade20kVal
    ```

### Demo

We support demo for image and image folder using `python run.py --type demo --cfg_file /path/to/yaml_file demo_path /path/to/image ct_score 0.3`.

For example:

```
python run.py --type demo --cfg_file configs/sbd_snake.yaml demo_path demo_images ct_score 0.3
# or
python run.py --type demo --cfg_file configs/city_snake.yaml demo_path demo_images/munster_000048_000019_leftImg8bit.png ct_score 0.3
```

If setup correctly, the output will be saved at `$ROOT/demo_out/` and look like

![demo](assets/city_vis.png)



## Training

The training parameters can be found in [project_structure.md](project_structure.md).

### Training on Cityscapes

```
python train_net.py --cfg_file configs/city_snake.yaml model rcnn_snake det_model rcnn_det
```

### Training on Kins

```
python train_net.py --cfg_file configs/kins_snake.yaml model kins_snake
```

### Training on Sbd

```
python train_net.py --cfg_file configs/sbd_snake.yaml model sbd_snake
```

### Training on COCO

```
python train_net.py --cfg_file configs/coco_snake.yaml model coco_snake
```

### Training on ADE20K

```
python train_net.py --cfg_file configs/ade20k.yaml model ade20k_snake
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{PolySnake2023,
  title={Recurrent Generic Contour-based Instance Segmentation with Progressive Learning},
  author={Feng, Hao and Zhou Keyi and Zhou, Wengang and Yin, Yufei and Deng, Jiajun and Sun, Qi and Li, Houqiang},
  booktitle={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024}
}
```

## Acknowledgement
Our work benefits a lot from [DeepSnake](https://github.com/zju3dv/snake) and [E2EC](https://github.com/zhang-tao-whu/e2ec). Thanks for their wonderful works.
