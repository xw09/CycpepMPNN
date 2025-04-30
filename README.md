## HighMPNN
This is the official implementation for the paper titled 'HighMPNN: A Graph Neural Network Approach for Structure-Constrained Cyclic Peptide Sequence Design'.
![alt text](Figure1.tif)
## Set up
For example to make a conda environment to run HighMPNN
```shell
conda create --name mlfold
source activate mlfold
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Detailed code of ProteinMPNN can be found in https://github.com/dauparas/ProteinMPNN

For more information about the cyclic peptide structure prediction model HighFold, see https://github.com/hongliangduan/HighFold.git
## Run
```shell
cd ./HighMPNN/script
bash run_test.sh
```