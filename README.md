# DTSM: Toward Dense Table Structure Recognition with Text Query Encoder and Adjacent Feature Aggregator
## Codes and data for the system presented in "DTSM: Toward Dense Table Structure Recognition with Text Query Encoder and Adjacent Feature Aggregator"

## Dataset
DenseTab is available at [here](https://drive.google.com/file/d/1WYyM_HyfyQ5tHakjUe72zUsAouvzPRV0/view?usp=drive_link).
After downloading the dataset locally, please modify your data paths by reprocessing the data with the scripts in the train_tool and test_tool directories of the zip archive.

## Requirements
- CUDA 11.7
- torch 1.13.0
- torchvision 0.14.0
- apted
- Distance
- lxml
- numpy
- opencv-python
- pandas
- Pillow
- Polygon3
- PyMuPDF
- scipy
- tqdm

## Training


Please change your data path and save path in libs/configs/defauli.py and execute 
```shell
bash runner/dist_train.sh
```
The experimental results may be a little better than those shown in the paper because we corrected some problem in DenseTab regarding the labelling.

## Credits
Parts of our code is based on:
https://github.com/ZZR8066/SEM

## Reference
Chen X., Chen B., Qu C., Peng D., Liu C., and Jin L. - International Conference on Document Analysis and Recognition (2024)
=======
# DTSM
Code and data for the paper: DTSM: Toward Dense Table Structure Recognition with Text Query Encoder and Adjacent Feature Aggregator
