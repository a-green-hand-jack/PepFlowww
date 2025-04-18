import math
import torch
from Bio import PDB
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict

from pepflow.modules.protein.constants import (AA, max_num_heavyatoms, max_num_hydrogens,
                        restype_to_heavyatom_names, 
                        restype_to_hydrogen_names,
                        BBHeavyAtom, non_standard_residue_substitutions)

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def _get_residue_heavyatom_info(res: Residue):
    """提取单个残基的重原子信息（坐标、存在掩码、B因子）
    
    Args:
        res (Bio.PDB.Residue): BioPython 的残基对象
    
    Returns:
        tuple: 
            - pos_heavyatom: 重原子坐标张量 [max_num_heavyatoms, 3]
            - mask_heavyatom: 原子存在掩码 [max_num_heavyatoms]
            - bfactor_heavyatom: B因子张量 [max_num_heavyatoms]
    """
    # 初始化输出张量（预设最大原子数）
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    
    # 获取残基类型对应的标准原子名列表
    restype = AA(res.get_resname())  # 转换为AA枚举类型
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '':  # 跳过空原子名（占位符）
            continue  
        
        # 如果当前原子存在于残基中
        if atom_name in res:
            # 记录原子坐标（转换为torch张量）
            pos_heavyatom[idx] = torch.tensor(
                res[atom_name].get_coord().tolist(),
                dtype=pos_heavyatom.dtype
            )
            mask_heavyatom[idx] = True  # 标记原子存在
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()  # 记录B因子
    
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom


def _get_residue_hydrogen_info(res: Residue):
    pos_hydrogen = torch.zeros([max_num_hydrogens, 3], dtype=torch.float)
    mask_hydrogen = torch.zeros([max_num_hydrogens, ], dtype=torch.bool)
    restype = AA(res.get_resname())

    for idx, atom_name in enumerate(restype_to_hydrogen_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_hydrogen[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_hydrogen.dtype)
            mask_hydrogen[idx] = True

    return pos_hydrogen, mask_hydrogen


def parse_pdb(path, model_id=0, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path, model_id, assembly_id=0, unknown_threshold=1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if '_pdbx_struct_assembly_gen.asym_id_list' not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)
    else:
        assemblies = [tuple(chains.split(',')) for chains in mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']]
        label_to_auth = {}
        for label_asym_id, auth_asym_id in zip(mmcif_dict['_atom_site.label_asym_id'], mmcif_dict['_atom_site.auth_asym_id']):
            label_to_auth[label_asym_id] = auth_asym_id
        model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
        return parse_biopython_structure(model_real)


def parse_biopython_structure(entity, unknown_threshold=1.0):
    """
    将 BioPython 的 PDB/MMCIF 结构对象解析为标准化张量格式
    
    Args:
        entity: BioPython 的结构对象 (Model/Chain/Residue 或其组合)
        unknown_threshold: 允许的未知氨基酸比例阈值 (超过则返回None)
    
    Returns:
        tuple: (data, seq_map)
            - data: EasyDict 包含解析后的张量数据
            - seq_map: 字典 {(chain_id, resseq, icode): 残基索引}
    """
    # 1. 展开所有链并排序
    chains = Selection.unfold_entities(entity, 'C')  # 获取所有链
    chains.sort(key=lambda c: c.get_id())  # 按链ID排序

    # 2. 初始化数据结构
    data = EasyDict({
        'chain_id': [], 'chain_nb': [],      # 链标识符和编号
        'resseq': [], 'icode': [], 'res_nb': [],  # PDB残基编号/插入码/内部连续编号
        'aa': [],                           # 氨基酸类型 (AA枚举值)
        'pos_heavyatom': [], 'mask_heavyatom': [],  # 重原子坐标和存在掩码
        # 注释掉的氢原子和B因子相关字段
    })

    # 3. 定义张量转换方法
    tensor_types = {
        'chain_nb': torch.LongTensor,
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,  # 堆叠为 [N_res, max_num_heavyatoms, 3]
        'mask_heavyatom': torch.stack,  # 堆叠为 [N_res, max_num_heavyatoms]
    }

    count_aa, count_unk = 0, 0  # 统计氨基酸和未知类型数量

    # 4. 遍历处理每个残基
    for i, chain in enumerate(chains):
        seq_this = 0  # 当前链的内部连续编号
        residues = Selection.unfold_entities(chain, 'R')  # 获取链中所有残基
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))  # 按(resseq, icode)排序

        for res in residues:
            resname = res.get_resname()
            # 跳过非标准氨基酸或缺少主链原子的残基
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue

            restype = AA(resname)  # 转换为枚举类型
            count_aa += 1
            if restype == AA.UNK:  # 统计未知类型
                count_unk += 1
                continue

            # 5. 记录链信息
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # 6. 记录氨基酸类型
            data.aa.append(restype)

            # 7. 提取重原子信息 (坐标/掩码/B因子)
            pos_heavyatom, mask_heavyatom, _ = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # 8. 生成连续残基编号 (处理缺失残基)
            resseq_this = res.get_id()[1]  # PDB中的残基编号
            icode_this = res.get_id()[2]   # 插入码
            if seq_this == 0:
                seq_this = 1
            else:
                # 通过CA-CA距离判断是否连续
                prev_CA = data.pos_heavyatom[-2][BBHeavyAtom.CA]
                curr_CA = data.pos_heavyatom[-1][BBHeavyAtom.CA]
                d_CA_CA = torch.linalg.norm(prev_CA - curr_CA, ord=2).item()
                
                if d_CA_CA <= 4.0:  # 正常肽键距离
                    seq_this += 1
                else:  # 处理缺失残基
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            # 记录残基编号信息
            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    # 9. 检查有效残基数量和未知比例
    if len(data.aa) == 0 or (count_unk / count_aa) >= unknown_threshold:
        return None, None

    # 10. 创建残基索引映射表
    seq_map = {
        (chain_id, resseq, icode): i 
        for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode))
    }

    # 11. 转换为PyTorch张量
    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])

    return data, seq_map
def get_fasta_from_pdb(pdb_file):
    parser = PDBParser()
    seq_dic = {}
    structure = parser.get_structure("structure_name", pdb_file)

    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if AA.is_aa(residue.get_resname()):
                    if residue.get_resname() == 'UNK':
                        sequence += 'X'
                    else:
                        sequence += PDB.Polypeptide.three_to_one(non_standard_residue_substitutions[residue.get_resname()])
            seq_dic[chain.id] = sequence

    return seq_dic

# def get_fasta_from_pdb(pdb_file):
#     parser = PDBParser()
#     structure = parser.get_structure("pdb", pdb_file)
    
#     fasta_sequence = ""
#     for chain in structure.get_chains():
#         for residue in chain.get_residues():
#             if residue.get_resname() in seq1(''):
#                 fasta_sequence += seq1(residue.get_resname())
    
#     return fasta_sequence


