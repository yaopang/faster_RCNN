# Train Py-Faster-RCNN with a Gaze Dataset

This tutorial is a fine-tuned clone of [zeyuanxy's one](https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train) for the [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) code.

We will illustrate how to train Py-Faster-RCNN on another dataset in the following steps, and we will take the **gaze database from RadLabs** as the example dataset.

## Clone py-faster-rcnn repository
The current tutorial need you to have clone and tested the regular py-faster-rcnn repository from rbgirshick.
```sh
$ git clone https://github.com/rbgirshick/py-faster-rcnn
```
We will refer to the root directory with $PY_FASTER_RCNN.

You will also need to follow the installation steps from [the original py-faster-rcnn readme](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md)

## Build the train-set

### Labeling

Options to label your images include:
1) Matlab
or
2) ```https://github.com/tzutalin/labelImg```

### Format the Dataset

But we will use this common architecture for every dataset in $PY_FASTER_RCNN/data
```
gaze_devkit/
|-- data/
    |-- Annotations/
         |-- *.txt (Annotation files)
    |-- Images/
         |-- *.png (Image files)
    |-- ImageSets/
         |-- train.txt
```

A simple way to achieve it is to use symbolic links:
(this is only an example for training, some refactoring will be needing in order to use the testset properly)
```sh
$ cd $PY_FASTER_RCNN/data
$ mkdir gaze_devkit/
$ mkdir gaze_devkit/data/
$ ln -s <path/of/gaze/database>/Annotations/ gaze_devkit/data/Annotations
$ ln -s <path/of/gase/database>/Images/ gaze_devkit/data/Images
```

Now we need to write `train.txt` that contains all the names(without extensions) of images files that will be used for training.
Basically with the following:
```sh
$ cd $PY_FASTER_RCNN/data/gaze_devkit/data/
$ mkdir ImageSets
$ ls Annotations/ -m | sed s/\\s/\\n/g | sed s/.txt//g | sed s/,//g > ImageSets/train.txt
```

### Add lib/datasets/yourdatabase.py
You need to add a new python file describing the dataset we will use to the directory `$PY_FASTER_RCNN/lib/datasets`, see [inria.py](https://github.com/deboc/py-faster-rcnn/blob/master/lib/datasets/inria.py). Then the following steps should be taken.
  - Modify `self._classes` in the constructor function to fit your dataset.
  - Be careful with the extensions of your image files. See `image_path_from_index` in `gaze.py`.
  - Write the function for parsing annotations. See `_load_gaze_annotation` in `gaze.py`.
  - Do not forget to add `import` syntaxes in your own python file and other python files in the same directory.

### Update lib/datasets/factory.py

Then you should modify the [factory.py](https://github.com/deboc/py-faster-rcnn/blob/master/lib/datasets/factory.py) in the same directory. For example, to add **gaze database**, we should add

```py
from datasets.gaze import gaze
gaze_devkit_path = '$PY_FASTER_RCNN/data/gaze_devkit'
for split in ['train', 'val']:
    name = '{}_{}'.format('gaze', split)
    __sets[name] = (lambda split=split: gaze(split, gaze_devkit_path))
```
**NB** : $PY_FASTER_RCNN must be replaced by its actual value !

## Adapt the network model

For example, if you want to use the model **VGG_CNN_M_1024** with alternated optimizations, then you should adapt the solvers in `$PY_FASTER_RCNN/models/VGG_CNN_M_1024/faster_rcnn_alt_opt/`

```sh
$ cd $PY_FASTER_RCNN/models/
$ mkdir gaze_model/
$ cp -r pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/ gaze_model/
```

It mainly concerns with the number of classes you want to train. Let's assume that the number of classes is C (do not forget to count the `background` class). Then you should 
  - Modify num_classes in 'RoIDataLayer' layer to 'C'
  - Modify `num_output` in the `cls_score` layer to 'C'
  - Modify `num_output` in the `bbox_pred` layer to '4 * C'

In our case we have 12 classes (including background):
```sh
$ grep 12 models/gaze_model/faster_rcnn_alt_opt/*.pt
faster_rcnn_test.pt:    num_output: 12
stage1_fast_rcnn_train.pt:    param_str: "'num_classes': 12"
stage1_fast_rcnn_train.pt:    num_output: 12
stage1_rpn_train.pt:    param_str: "'num_classes': 12"
stage2_fast_rcnn_train.pt:    param_str: "'num_classes': 12"
stage2_fast_rcnn_train.pt:    num_output: 12
stage2_rpn_train.pt:    param_str: "'num_classes': 12"

$ grep 48 models/gaze_model/faster_rcnn_alt_opt/*.pt
faster_rcnn_test.pt:    num_output: 48
stage1_fast_rcnn_train.pt:    num_output: 48
stage2_fast_rcnn_train.pt:    num_output: 48
```

## Build config file

The $PY_FASTER_RCNN/models folder must be specified by a config file as in [faster_rcnn_alt_opt.yml](https://github.com/deboc/py-faster-rcnn/blob/master/help/faster_rcnn_alt_opt.yml)
```sh
$ echo 'MODELS_DIR: "$PY_FASTER_RCNN/models"' >> config.yml
```
**NB** : $PY_FASTER_RCNN must be replaced by its actual value !

## Launch the training

In the directory $PY_FASTER_RCNN, run the following command in the shell.

```sh
$ ./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name gaze_model --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb gaze_train 
```

Where:    
>--net_name is the folder name in $PY_FASTER_RCNN/models    
>    (nb: the train_faster_rcnn_alt_opt.py script will automatically look into the /faster_rcnn_alt_opt/ subfolder for the .pt files)    
>--weights is the optional location of pretrained weights in .caffemodel    
>--imdb is the full name of the database as specified in the lib/datasets/factory.py file    
>    (nb: dont forget to add the test/train suffix !)    

If you're sshing to the server, use ```screen``` to keep process running if connection drops unexpectedly.

# Testing Py-Faster-RCNN 

Run this line to test your model:
```
$ cd $PY_FASTER_RCNN/
$ ./tools/test_net.py --gpu 0 --def models/gaze_model_end2end/test.prototxt --net output/faster_rcnn_end2end/train/vgg_cnn_m_1024_gaze_end2end_iter_20000.caffemodel --imdb gaze_val --cfg experiments/cfgs/faster_rcnn_end2end.yml 
```


