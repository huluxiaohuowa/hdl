# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/chemical_tools/query_info.py
# 说明：化学结构文件与分子查询工具
# 模块功能：按名称查化合物的 SMILES/CAS/同义名（先查 PostgreSQL 映射表，未命中再走 PubChem 与 CIR），并把查询结果回写数据库
import re

import cirpy
import pubchempy as pcp
# from rdkit import Chem
import molvs as mv
from psycopg import sql

from hdl.utils.database_tools.connect import connect_by_infofile


def query_from_cir(query_name: str):
    """
    用 CIR（Chemical Identifier Resolver，cirpy）按名称联网查化合物信息。
    Returns:
        (smiles, cas_list, name_list)：molvs 标准化后的 SMILES（失败时为原始字符串或 None）、
        CAS 号列表、同义名称列表；查不到的字段为空列表。
    """
    smiles = None
    # cas_list = []
    # name_list = []
    
    # 分别请求 CAS 号、名称、SMILES；返回 None 或标量时统一成列表
    cas_list = cirpy.resolve(query_name, 'cas')
    if cas_list is None or not cas_list:
        cas_list = []
    if isinstance(cas_list, str):
        cas_list = [cas_list]

    name_list = cirpy.resolve(query_name, 'names')
    if name_list is None or not name_list:
        name_list = []
    if isinstance(name_list, str):
        name_list = [name_list]

    smiles = cirpy.resolve(query_name, 'smiles')
    # 用 molvs 做结构标准化（standardization）；失败时保留 cir 返回的原始字符串
    try:
        smiles = mv.standardize_smiles(smiles)
    except Exception as e:
        print(e)

    return smiles, cas_list, name_list


def query_from_pubchem(query_name: str):
    """
    用 PubChem（pubchempy）按化合物名称检索，取首个结果的 canonical SMILES。
    Returns:
        (smiles, cas_list, name_list)：标准化 SMILES、从同义词正则提取的 CAS 号列表、全部同义词列表；
        无结果时返回 (None, 空集合, 空集合)。
    """
    results = pcp.get_compounds(query_name, 'name')
    smiles = None
    name_list = set()
    cas_list = set()

    if any(results):
        try:
            smiles = mv.standardize_smiles(results[0].canonical_smiles)
        except Exception as e:
            # 标准化失败时退回 PubChem 原始 canonical SMILES
            smiles = results[0].canonical_smiles
            print(smiles)
            print(e)
        for compound in results:
            # 汇总所有命中记录的 synonym；re.match 只从字符串开头比对，挑出形如 1234-56-7 的 CAS 号
            name_list.update(set(compound.synonyms))
            for syn in compound.synonyms:
                match = re.match('(\d{2,7}-\d\d-\d)', syn)
                if match:
                    cas_list.add(match.group(1))
        
        cas_list = list(cas_list)
        name_list = list(name_list)

    return smiles, cas_list, name_list


def query_a_compound(
    query_name: str,
    connect_info: str,
    by: str = 'name',
    log_file: str = './err.log'
):
    """
    按名称查一个化合物的数据库主键 fei：先查本地 name_maps 表，查不到就转 PubChem/CIR 联网查询，
    再把结果回写 compounds / cas_maps / name_maps 三张表。
    Args:
        query_name: 化合物名称，先转小写再匹配（函数内强制按 name 查询，忽略 by 的取值）。
        connect_info: 传给 connect_by_infofile 的数据库连接信息文件路径。
        by: 查询字段名，用于拼出表名 <by>_maps；当前实现固定改写为 'name'。
        log_file: 联网也查不到时追加记录该名称的日志文件。
    Returns:
        fei（化合物的 FEI/数据库标识字符串）；本地表和联网均查不到时返回 None（且不关闭连接）。
    """
    fei = None
    found = False  

    query_name = query_name.lower()
    
    by = 'name'
    table = by + '_maps'
    # query_name = 'adipic acid'
    # 用 psycopg.sql 安全拼装表名/字段名，避免拼接字符串引入注入
    query = sql.SQL(
        "select fei from {table} where {by} = %s"
    ).format(
        table=sql.Identifier(table),
        by=sql.Identifier(by)
    )
    conn = connect_by_infofile(connect_info)
    
    # 先查名称映射表，命中即直接返回 fei
    cur = conn.execute(query, [query_name]).fetchone()

    if cur is not None:
        fei = cur[0]
        found = True
        return fei
 
    # 本地未命中：先 PubChem，再 CIR，任一拿到 SMILES 即视为查到
    if not found:
        try:
            smiles, cas_list, name_list = query_from_pubchem(query_name) 
        except Exception as e:
            print(e)
            smiles, cas_list, name_list = None, [], []
        if smiles is not None:
            found = True
        else:
            try:
                smiles, cas_list, name_list = query_from_cir(query_name)
            except Exception as e:
                print(e)
                smiles, cas_list, name_list = None, [], []
            if smiles is not None:
                found = True
    
    # 两路都查不到：把名称写进错误日志后返回 None
    if not found:
        with open(log_file, 'a') as f:
            f.write(query_name)
            f.write('\n')
        return
        # raise ValueError('给的啥破玩意儿查都查不着！')
    else:
        # 用 SMILES 反查 compounds 表是否已登记
        query_compound = sql.SQL(
            "select fei from compounds where smiles = %s"
        )
        cur = conn.execute(query_compound, [smiles]).fetchone()
        if cur is not None:
            # compounds 表已有该 SMILES，直接复用其 fei
            fei = cur[0]
        elif any(cas_list):
            # 新化合物：以首个 CAS 号作为 fei，插入 compounds（冲突则跳过）
            fei = cas_list[0]
            insert_compounds_sql = sql.SQL(
                "INSERT INTO compounds (fei, smiles) VALUES (%s, %s) ON CONFLICT (fei) DO NOTHING"
            )
            conn.execute(insert_compounds_sql, [fei, smiles])
        # 把全部 CAS 号与名称（小写）补写进两张映射表，单条失败只打印不影响其余
        for cas in cas_list:
            insert_cas_map_sql = sql.SQL(
                "INSERT INTO cas_maps (fei, cas) VALUES (%s, %s) ON CONFLICT (cas) DO NOTHING"
            )
            try:
                conn.execute(insert_cas_map_sql, [fei, cas])
            except Exception as e:
                print(e)
        for name in name_list:
            insert_name_map_sql = sql.SQL(
                "INSERT INTO name_maps (fei, name) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING"
            ) 
            try:
                conn.execute(insert_name_map_sql, [fei, name.lower()])
            except Exception as e:
                print(e)

    # 提交回写的记录并断开连接（前面按名称命中时已提前 return，不走这里）
    conn.commit()
    conn.close()

    return fei
