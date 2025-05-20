import json
import numpy as np
import os
import os.path
import os.path
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import Superimposer
from Bio.PDB.Atom import Atom
from concurrent.futures import ProcessPoolExecutor
from torch import optim
from torch.utils.data import DataLoader

def extract_plddt_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if 'plddt' in data:
            plddt_list = data['plddt']
            return plddt_list
        else:
            raise KeyError("The key 'plddt' does not exist in the JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def huber_loss(y_pred, y_true, delta_threshold=1):
    delta = torch.abs(y_pred - y_true)

    loss = torch.where(delta <= delta_threshold,
                       0.5 * delta ** 2,  
                       delta_threshold * (delta - 0.5 * delta_threshold))  
    return loss.mean()


def combined_loss_fn(loss_av_smoothed, loss_other, if_mon, alpha):

    stru_loss = huber_loss(loss_other, torch.zeros_like(loss_other))  
    if if_mon == 1:
        '单体/20'
        # combined_loss = loss_av_smoothed / 20 + stru_loss*0.1
        combined_loss = loss_av_smoothed / 20+ stru_loss*alpha
        # combined_loss = loss_av_smoothed
    else:
         combined_loss = loss_av_smoothed + stru_loss*alpha
    # combined_loss = loss_av_smoothed + stru_loss / 20*0.5

    return combined_loss


def find_file_with_name(folder_paths, search_strings):
    file_path_pdb = 'no such file'
    for root, dirs, files in os.walk(folder_paths):
        for file in files:

            if search_strings in file:
                file_path_pdb = os.path.join(root, file)
                return file_path_pdb 

    print(f"未找到包含 '{folder_paths}/{search_strings}' 的文件。")
    return 'no such file'


def check_and_clear_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        item_count = len(os.listdir(folder_path))

        if item_count > 3:
            clear_folder(folder_path)
    else:
        print(f"文件夹 '{folder_path}' 不存在。")


def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的目录。")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            if not item.endswith('.a3m'):
                os.remove(item_path)

        elif os.path.isdir(item_path):
            if item not in ['sequence_', 'sequence_env']:
                shutil.rmtree(item_path)
    print('清空', folder_path, ',保留MSA文件')


def highfold(seq, seq_index, input_folder, output_folder, num_of_chains):

    fasta_filename = os.path.join(input_folder, f"sequence_{seq_index}.fasta")
    save_seq_to_fasta(seq, fasta_filename)

    conda_setup = "eval \"$(/home/light/mambaforge/bin/conda shell.bash hook)\""  
    # conda_activate_cmd = "conda activate highfold_xw" # recycle=0
    conda_activate_cmd = "conda activate highfold_multi_ligand" # recycle=3

    # HighFold
    base_command = "colabfold_batch --templates --model-type alphafold2_multimer_v3"

    output_file = os.path.join(output_folder, f"{seq_index}_AF2_output")

    full_command = f"{base_command} {fasta_filename} {output_file} --flag-cyclic-peptide 1 --flag-nc 1"

    full_cmd = f"bash -c '. ~/.bashrc; {conda_setup} && {conda_activate_cmd} && {full_command}'"

    try:
        result = subprocess.run(full_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"命令执行成功：\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败：\n{e.stderr}")


def save_seq_to_fasta(seq, filename):
    try:
        with open(filename, 'w') as file:
            file.write(f">sequence\n{seq}\n")
        print(f"序列已成功保存到: {filename}")
    except Exception as e:
        print(f"保存序列时发生错误: {e}")
