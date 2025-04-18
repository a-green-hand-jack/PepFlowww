"""pep-rec dataset"""

import os
import logging
import joblib
import pickle
import lmdb
from Bio import PDB
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from pepflow.modules.protein.parsers import parse_pdb
from pepflow.modules.common.geometry import *
from pepflow.modules.protein.constants import *
from pepflow.utils.data import (
    mask_select_data,
    find_longest_true_segment,
    PaddingCollate,
)
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from easydict import EasyDict

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist

from pepflow.utils.misc import load_config
from pepflow.utils.train import recursive_to

from models_con.torsion import get_torsion_angle

import torch

from pepflow.modules.protein.writers import save_pdb

# bind_dic = torch.load("/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/misc/affinity_dict.pt")

# testset
names = []
with open(
    "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pepflowww/Data/names.txt",
    "r",
) as f:
    for line in f:
        names.append(line.strip())

def preprocess_structure(task):
    """预处理单个蛋白质-肽段复合物结构任务
    
    Args:
        task (dict): 包含以下键值：
            - "id": 任务标识符（PDB文件名）
            - "pdb_path": 包含肽段(pocket.pdb)和受体(peptide.pdb)的目录路径
    
    Returns:
        dict: 整合后的结构数据，包含受体和肽段信息
    """
    try:
        # 检查ID是否在允许的命名列表中（未显示的`names`变量）
        if task["id"] in names:
            raise ValueError(f"{task['id']} not in names")

        pdb_path = task["pdb_path"]
        
        # ----------------------------
        # 1. 处理肽段(peptide)
        # ----------------------------
        # 解析肽段PDB文件（第一个返回的结构）
        pep = parse_pdb(os.path.join(pdb_path, "peptide.pdb"))[0]  
        
        # 计算肽段CA原子的几何中心（质心）
        center = torch.sum(
            pep["pos_heavyatom"][  # 重原子坐标 [L, 14, 3]
                pep["mask_heavyatom"][:, BBHeavyAtom.CA],  # 取CA原子掩码
                BBHeavyAtom.CA
            ],
            dim=0,
        ) / (torch.sum(pep["mask_heavyatom"][:, BBHeavyAtom.CA]) + 1e-8)
        
        # 将肽段坐标归一化到质心坐标系
        pep["pos_heavyatom"] = pep["pos_heavyatom"] - center[None, None, :]
        
        # 计算肽段二面角（平移坐标后计算）
        pep["torsion_angle"], pep["torsion_angle_mask"] = get_torsion_angle(
            pep["pos_heavyatom"], pep["aa"]
        )
        
        # 检查肽段长度限制（3-25个残基）
        if len(pep["aa"]) < 3 or len(pep["aa"]) > 25:
            raise ValueError("peptide length not in [3,25]")

        # ----------------------------
        # 2. 处理受体(pocket)
        # ----------------------------
        rec = parse_pdb(os.path.join(pdb_path, "pocket.pdb"))[0]
        # 受体坐标同样归一化到肽段质心坐标系
        rec["pos_heavyatom"] = rec["pos_heavyatom"] - center[None, None, :]
        
        # 计算受体二面角
        rec["torsion_angle"], rec["torsion_angle_mask"] = get_torsion_angle(
            rec["pos_heavyatom"], rec["aa"]
        )
        
        # 受体链编号+1（可能与肽段链号区分）
        rec["chain_nb"] += 1

        # ----------------------------
        # 3. 合并数据
        # ----------------------------
        data = {}
        data["id"] = task["id"]  # 保留原始ID
        
        # 生成掩码：受体部分为0，肽段部分为1
        data["generate_mask"] = torch.cat(
            [torch.zeros_like(rec["aa"]), torch.ones_like(pep["aa"])], 
            dim=0
        ).bool()
        
        # 合并受体和肽段的所有特征
        for k in rec.keys():
            if isinstance(rec[k], torch.Tensor):
                # 张量沿序列维度拼接（如原子坐标、氨基酸类型等）
                data[k] = torch.cat([rec[k], pep[k]], dim=0)
            elif isinstance(rec[k], list):
                # 列表直接合并
                data[k] = rec[k] + pep[k]
            else:
                raise ValueError(f"Unknown type of {rec[k]}")
                
        return data
    except Exception as e:
        # 打印错误信息并返回None（外层会跳过该数据）
        print(f"Error processing {task['id']}: {str(e)}")
        return None
    


class PepDataset(Dataset):
    """用于加载和处理蛋白质结构数据的 PyTorch Dataset 类。"""

    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB，LMDB 数据库的最大存储空间

    def __init__(
        self,
        structure_dir="./Data/PepMerge_new/",  # 原始 PDB 文件存储目录
        dataset_dir="./Data/",                 # 数据集根目录（LMDB 缓存存储位置）
        name="pep",                            # 数据集名称（用于生成缓存文件名）
        transform=None,                        # 可选的预处理变换函数
        reset=False,                           # 是否重置缓存（重新预处理所有数据）
    ):
        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.name = name

        self.db_conn = None  # LMDB 数据库连接
        self.db_ids = None    # 存储所有已处理数据的 ID 列表
        self._load_structures(reset)  # 初始化时加载或预处理数据

    @property
    def _cache_db_path(self):
        """返回 LMDB 缓存文件的完整路径。"""
        return os.path.join(self.dataset_dir, f"{self.name}_structure_cache.lmdb")

    def _connect_db(self):
        """连接 LMDB 数据库并加载所有数据的 ID 列表。"""
        self._close_db()  # 确保关闭之前的连接
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,     # 不创建新数据库
            subdir=False,     # 缓存文件是单独文件（非子目录）
            readonly=True,    # 只读模式
            lock=False,      # 禁用文件锁（多进程安全）
            readahead=False,  # 禁用预读（优化性能）
            meminit=False,    # 禁用内存初始化（优化性能）
        )
        with self.db_conn.begin() as txn:
            # 获取所有键（即已处理数据的 ID）
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        """关闭 LMDB 数据库连接并清空 ID 列表。"""
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def _load_structures(self, reset):
        """加载或预处理蛋白质结构数据。
        
        Args:
            reset (bool): 是否强制重新预处理所有数据。
        """
        all_pdbs = os.listdir(self.structure_dir)  # 获取所有原始 PDB 文件名

        if reset:
            # 重置模式：删除现有缓存并重新处理所有数据
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()
            todo_pdbs = all_pdbs  # 需要处理的所有文件
        else:
            # 非重置模式：仅处理未缓存的数据
            if not os.path.exists(self._cache_db_path):
                todo_pdbs = all_pdbs
            else:
                todo_pdbs = []  # 注释部分代码未启用（默认处理所有数据）

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)  # 预处理数据

    def _preprocess_structures(self, pdb_list):
        """并行预处理 PDB 文件并存储到 LMDB 数据库。
        
        Args:
            pdb_list (list): 需要预处理的 PDB 文件名列表。
        """
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            tasks.append(
                {
                    "id": pdb_fname,      # 使用文件名作为唯一 ID
                    "pdb_path": pdb_path,  # PDB 文件路径
                }
            )

        # 使用 joblib 并行预处理（使用一半 CPU 核心）
        data_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc="Preprocess")
        )

        # 将预处理结果写入 LMDB 数据库
        db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=True,      # 允许创建新数据库
            subdir=False,
            readonly=False,   # 可写模式
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc="Write to LMDB"):
                if data is None:  # 跳过预处理失败的数据
                    continue
                ids.append(data["id"])
                txn.put(data["id"].encode("utf-8"), pickle.dumps(data))  # 序列化存储

    def __len__(self):
        """返回数据集大小（通过 LMDB 中的 ID 数量计算）。"""
        self._connect_db()  # 确保数据库已连接
        return len(self.db_ids)

    def __getitem__(self, index):
        """根据索引获取预处理后的数据。
        
        Args:
            index (int): 数据索引。
        Returns:
            dict: 预处理后的数据（可能经过 transform 处理）。
        """
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))  # 反序列化数据
        if self.transform is not None:
            data = self.transform(data)  # 应用额外变换
        return data
    


if __name__ == "__main__":
    device = "cuda:1"
    config, cfg_name = load_config("./configs/learn/learn_all.yaml")
    dataset = PepDataset(
        structure_dir="./Data/PepMerge_new/",
        dataset_dir="/Data/Fixed Data",
        name="pep_pocket_test",
        transform=None,
        reset=True,
    )
    print(len(dataset))
    print(dataset[0])

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=PaddingCollate(eight=False),
    )

    batch = next(iter(dataloader))
    print(batch["torsion_angle"].shape)
    print(batch["torsion_angle_mask"].shape)
