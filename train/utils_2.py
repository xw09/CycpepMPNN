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

def caculate_ca_rmsd(X_cleaned_np, X_pdb_np):

    def convert_to_atoms(coord_array, atom_indices=[0], atom_names=['CA']):
        atom_list = []
        for res_index, residue in enumerate(coord_array):
            for atom_index, coord in enumerate(residue):
                # 仅选择指定的原子（N, CA, C, O）
                if atom_index in atom_indices:
                    name = atom_names[atom_indices.index(atom_index)]
                    atom = Atom(name, coord, 0, 1, '', name + str(res_index), res_index)
                    atom_list.append(atom)
        return atom_list

    '对齐蛋白质计算多肽RMSD'

    # 生成用于 RMSD 计算的 Bio.PDB Atom 对象列表，仅包括多肽部分的 CA 原子
    fixed_atoms_peptide = convert_to_atoms(X_cleaned_np, atom_indices=[0], atom_names=['CA'])
    moving_atoms_peptide = convert_to_atoms(X_pdb_np, atom_indices=[0], atom_names=['CA'])

    # 创建Superimposer对象，并对齐多肽
    sup = Superimposer()

    # 计算多肽的 CA 原子的 RMSD
    sup.set_atoms(fixed_atoms_peptide, moving_atoms_peptide)
    return sup.rms

def kabsch(P, Q):
    # if P.dim() > 2:
    #     P = P.view(-1, 3)  # 展平为 (n, 3)
    # if Q.dim() > 2:
    #     Q = Q.view(-1, 3)  # 展平为 (n, 3)
    # # Compute covariance matrix
    # C = torch.matmul(P.mT, Q)

    # # Singular Value Decomposition (SVD)
    # V, S, Vh = torch.linalg.svd(C.to(dtype=torch.float))

    # # Compute determinant to ensure a proper rotation (no reflection)
    # d = torch.det(torch.matmul(Vh.to(dtype=torch.double).mT, V.to(dtype=torch.double).mT))
    # D = torch.eye(3, device=P.device)
    # d = d.item()  # 转换为Python浮动类型
    # D[-1, -1] = d

    # # Compute the rotation matrix
    # R = torch.matmul(torch.matmul(Vh.mT, D), V.mT)
    # return R
    # Compute covariance matrix
    C = torch.mm(P.T, Q)
    
    # Singular Value Decomposition (SVD)
    U, S, V = torch.svd(C)  # Note: V is returned transposed in torch.svd()
    
    # Compute determinant to ensure a proper rotation (no reflection)
    d = torch.det(torch.mm(V, U.T))
    D = torch.eye(3, device=P.device)
    D[2, 2] = torch.sign(d)
    
    # Compute the rotation matrix
    U_final = torch.mm(torch.mm(V, D), U.T)
    return U_final


def superimpose_and_calculate_rmsd(coords1, coords2):
    # # Calculate centroids
    # P_centroid = torch.mean(coords1, dim=0)
    # Q_centroid = torch.mean(coords2, dim=0)

    # # Center coordinates
    # P_centered = coords1 - P_centroid
    # Q_centered = coords2 - Q_centroid

    # # Compute rotation matrix using Kabsch algorithm
    # U = kabsch(P_centered, Q_centered)

    # # Apply rotation matrix to Q_centered
    # Q_rotated = torch.matmul(Q_centered, U)

    # # Compute RMSD
    # # diff = P_centered[:, 1, :] - Q_rotated[:, 1, :]
    # # rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))

    # diff = P_centered[:, 1] - Q_rotated[:, 1]
    # rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2)))
    # return rmsd
    # Calculate centroids
    P_centroid = torch.mean(coords1, dim=0)
    Q_centroid = torch.mean(coords2, dim=0)
    
    # Center coordinates
    P_centered = coords1 - P_centroid
    Q_centered = coords2 - Q_centroid
    
    # Compute rotation matrix using Kabsch algorithm
    U = kabsch(P_centered, Q_centered)
    
    # Apply rotation matrix to Q_centered
    Q_rotated = torch.mm(Q_centered, U)
    
    # Compute RMSD
    diff = P_centered - Q_rotated
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=1)))
    return rmsd


# #将预测的二级结构X_pdb与真实二级结构X进行对比，得到RMSD
def calculate_peptide_rmsd(X, X_pre, num_seq, len_pro, len_pep=0):
    if num_seq == 1:
        '单体-主链N/CA/C/O原子rmsd'
        if X.size() == X_pre.size():
            X_true = X
        else:
            mask = (X != 0).any(dim=-1).all(dim=-2)  # 更新mask以保证所有原子均被考虑

            X_true = X[mask].view(1, -1, 4, 3)

        'RMSD计算'
        ca_true = X_true.squeeze(0)[:, 1, :]
        ca_pre = X_pre.squeeze(0)[:, 1, :]

        true = X_true.squeeze(0)
        pre = X_pre.squeeze(0)

        rmsd_pro_align = superimpose_and_calculate_rmsd(true, pre)

    else:
        '复合物-对齐蛋白算肽骨架'
        if X.size() == X_pre.size():
            X_true = X
        else:
            mask = (X != 0).any(dim=-1)

            X_true = X[mask].view(1, -1, 4, 3)

        'RMSD计算''X前面是肽链,后面是蛋白,X_pdb前面是蛋白,后面是肽链'
        ca_true = X_true.squeeze(0)[:, 1, :]  # 假设第二个原子是 CA 原子
        ca_pre = X_pre.squeeze(0)[:, 1, :]

        # 提取蛋白质部分（最后后15个）用于对齐
        X_true_protein = ca_true[-len_pro:, :]  # !!!!!
        # 提取蛋白质部分（前15个）用于对齐
        X_pre_protein = ca_pre[:len_pro, :]

        # 提取多肽部分（前5个）
        X_true_peptide = ca_true[:len_pep, :]  # !!!!!
        # 提取多肽部分（最后5个）
        X_pre_peptide = ca_pre[-len_pep:, :]

        # 第一步：对齐蛋白质部分
        U = kabsch(X_true_protein, X_pre_protein)

        # 第二步：将旋转矩阵应用于清理后的结构中的多肽部分
        X_true_peptide_aligned_ca = torch.matmul(
            X_true_peptide - torch.mean(X_true_peptide, dim=0), U
        ) + torch.mean(X_pre_peptide, dim=0)

        # 计算 CA 原子的 RMSD
        rmsd_pro_align = superimpose_and_calculate_rmsd(X_true_peptide_aligned_ca, X_pre_peptide)

    return rmsd_pro_align


def extract_plddt_from_json(file_path):
    try:
        # 打开并加载 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 提取 plddt 列表
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
                       0.5 * delta ** 2,  # L2损失（对于小误差）
                       delta_threshold * (delta - 0.5 * delta_threshold))  # L1损失（对于大误差）
    return loss.mean()


# 运用神经网络，将结构损失加入到总损失中
def combined_loss_fn(loss_av_smoothed, loss_other, if_mon, alpha):

    rmsd_loss = huber_loss(loss_other, torch.zeros_like(loss_other))  # 假设真实值为零
    if if_mon == 1:
        '单体/20'
        # combined_loss = loss_av_smoothed / 20 + rmsd_loss*0.1
        combined_loss = loss_av_smoothed / 20+ rmsd_loss*alpha
        # combined_loss = loss_av_smoothed
    else:
         combined_loss = loss_av_smoothed + rmsd_loss*alpha
    # combined_loss = loss_av_smoothed + rmsd_loss / 20*0.5

    return combined_loss


# 读取预测的pdb文件，并提取N,CA,C,O原子坐标表示特征
def find_file_with_name(folder_paths, search_strings):
    """
    在指定文件夹中查找包含特定字符串的文件，并返回文件路径。
    """
    file_path_pdb = 'no such file'
    for root, dirs, files in os.walk(folder_paths):
        for file in files:
            # 检查文件名中是否包含指定字符串
            if search_strings in file:
                file_path_pdb = os.path.join(root, file)
                return file_path_pdb  # 找到第一个匹配的文件就返回路径
    # 如果没有找到匹配文件，则返回 None
    print(f"未找到包含 '{folder_paths}/{search_strings}' 的文件。")
    return 'no such file'


def check_and_clear_folder(folder_path):
    """
    检查文件夹中的文件和文件夹数量，如果数量大于三，则执行清理操作。
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        item_count = len(os.listdir(folder_path))

        if item_count > 3:
            clear_folder(folder_path)
    else:
        print(f"文件夹 '{folder_path}' 不存在。")


def clear_folder(folder_path):
    """
    检查输出文件夹是否为空,保留MSA有关文件
    """
    # 检查文件夹是否为空
    # 确保提供的路径是一个目录
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的目录。")
        return

    # 遍历文件夹中的所有文件和子文件夹
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 如果是文件且不是 .a3m 文件，删除它
        if os.path.isfile(item_path):
            if not item.endswith('.a3m'):
                os.remove(item_path)

        # 如果是文件夹，检查文件夹名
        elif os.path.isdir(item_path):
            if item not in ['sequence_', 'sequence_env']:
                shutil.rmtree(item_path)
    print('清空', folder_path, ',保留MSA文件')


def highfold(seq, seq_index, input_folder, output_folder, num_of_chains):
    """
    将序列输入到 highfold 中，得到预测的二级结构
    """
    # 保存序列为 .fasta 文件
    fasta_filename = os.path.join(input_folder, f"sequence_{seq_index}.fasta")
    save_seq_to_fasta(seq, fasta_filename)

    # 激活 conda 环境的命令
    conda_setup = "eval \"$(/home/light/mambaforge/bin/conda shell.bash hook)\""  # 用于设置 conda 环境
    # conda_activate_cmd = "conda activate highfold_xw" # recycle=0
    conda_activate_cmd = "conda activate highfold_multi_ligand" # recycle=3

    # 指定命令的基本部分
    # AF2
    # base_command="colabfold_batch --templates --amber --model-type alphafold2_multimer_v3"

    # HighFold
    base_command = "colabfold_batch --templates --model-type alphafold2_multimer_v3"

    # 构建输出文件路径
    output_file = os.path.join(output_folder, f"{seq_index}_AF2_output")

    # HIGHFOLD 复合物命令
    if num_of_chains == 1:
        # HIGHFOLD单体
        full_command = f"{base_command} {fasta_filename} {output_file} --flag-cyclic-peptide 1 --flag-nc 1"
    else:
        # HIGHFOLD复合物
        full_command = f"{base_command} {fasta_filename} {output_file} --flag-cyclic-peptide 0 1 --flag-nc 0 1"

    # AF2
    # full_command = f"{base_command} {fasta_filename} {output_file}"

    # 打印并执行命令
    # print(f"准备执行命令：{full_command}")
    full_cmd = f"bash -c '. ~/.bashrc; {conda_setup} && {conda_activate_cmd} && {full_command}'"

    try:
        # 执行完整命令
        result = subprocess.run(full_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"命令执行成功：\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败：\n{e.stderr}")


def save_seq_to_fasta(seq, filename):
    # 保存序列到 .fasta 文件
    try:
        with open(filename, 'w') as file:
            file.write(f">sequence\n{seq}\n")
        print(f"序列已成功保存到: {filename}")
    except Exception as e:
        print(f"保存序列时发生错误: {e}")
