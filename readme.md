# YLOC

## Description
YLOC is a project designed for few-shot class incremental learning. 

The project utilizes various public datasets, including CIFAR-100, CUB-200, and MiniImageNet. The datasets can be found [here](https://github.com/icoz69/CEC-CVPR2021).

## Requirements
To run this project, ensure you have the following dependencies installed:

- PyTorch (>= version 1.1) and torchvision
- tqdm
- sklearn
- numpy



## Running Procedure
1. Download the datasets and environments
2. Pretrain the base model by running pretrain_base_model.sh. You need to add the dataset's directory, the location of the training log, and the name of the dataset.
3. After you finish pretraining the model, modify run_incremental.sh for incremental sessions. You need to add the dataset's directory, the location of the training log, the resume model, and the name of the dataset.


## Acknowledgements
We thank the following repositories for providing helpful components and functions in our work:


1. [C-FCIL](https://github.com/IBM/constrained-FSCIL)
2. [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
3. [SAVC](https://github.com/zysong0113/SAVC)



## License
This project is released under the MIT License.

