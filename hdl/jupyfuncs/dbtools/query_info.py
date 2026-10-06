# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dbtools/query_info.py
# 说明：数据库查询工具
import re

import cirpy
import pubchempy as pcp
import molvs as mv
from psycopg import sql

from .pg import connect_by_infofile


def query_from_cir(query_name: str):
    """走 CIR（Chemical Identifier Resolver，cirpy 客户端）做标识符解析：对同一个 query_name 依次请求 'cas'、'names'、'smiles' 三类结果（联网访问 CIR 服务），
    返回 None 或空结果的归一化成空列表、单个字符串结果包成单元素列表；SMILES 再交给 molvs 的 standardize_smiles 标准化成统一写法，标准化失败只打印异常、保留未标准化的原值；
    返回三元组 (smiles, cas_list, name_list)：查不到结构式时 smiles 仍为 None，但 cas_list/name_list 可能已有内容，调用方以 smiles 是否为 None 判定命中。"""
    smiles = None
    # cas_list = []
    # name_list = [host=172.20.0.5 dbname=pistachio port=5432 user=postgres password=woshipostgres]

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
    try:
        smiles = mv.standardize_smiles(smiles)
    except Exception as e:
        print(e)

    return smiles, cas_list, name_list


def query_from_pubchem(query_name: str):
    """走 PubChem（pubchempy）按化合物名查询：pcp.get_compounds(query_name, 'name') 以 name 字段检索，返回命中化合物对象列表（需联网）；
    有结果时取第一条的 canonical_smiles 并用 molvs 标准化，标准化抛异常则退回原始 canonical_smiles 并打印该 SMILES 与异常信息；
    再遍历所有记录的 synonyms 汇总为别名集合，并用正则（形如 2-7 位数字-2 位数字-1 位数字）从别名文本中筛出 CAS 号；
    返回 (smiles, cas_list, name_list)，两个列表由集合转成、顺序不保证；未命中时 smiles 为 None 且两者为空列表。"""
    results = pcp.get_compounds(query_name, 'name')
    smiles = None
    name_list = set()
    cas_list = set()

    if any(results):
        try:
            smiles = mv.standardize_smiles(results[0].canonical_smiles)
        except Exception as e:
            smiles = results[0].canonical_smiles
            print(smiles)
            print(e)
        for compound in results:
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
    fei = None
    found = False  

    if by != 'smiles':
        query_name = query_name.lower()
    
    by = 'name'
    table = by + '_maps'
    # query_name = 'adipic acid'
    query = sql.SQL(
        "select fei from {table} where {by} = %s"
    ).format(
        table=sql.Identifier(table),
        by=sql.Identifier(by)
    )
    conn = connect_by_infofile(connect_info)
    
    cur = conn.execute(query, [query_name]).fetchone()

    if cur is not None:
        fei = cur[0]
        found = True
        return fei
 
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
    
    if not found:
        with open(log_file, 'a') as f:
            f.write(query_name)
            f.write('\n')
        return
        # raise ValueError('给的啥破玩意儿查都查不着！')
    else:
        query_compound = sql.SQL(
            "select fei from compounds where smiles = %s"
        )
        cur = conn.execute(query_compound, [smiles]).fetchone()
        if cur is not None:
            fei = cur[0]
        elif any(cas_list):
            fei = cas_list[0]
            insert_compounds_sql = sql.SQL(
                "INSERT INTO compounds (fei, smiles) VALUES (%s, %s) ON CONFLICT (fei) DO NOTHING"
            )
            conn.execute(insert_compounds_sql, [fei, smiles])
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
                "INSERT INTO name_maps (fei, name) VALUES (%s, %s) ON CONFLICT (fei, name) DO NOTHING"
                # "INSERT INTO name_maps (fei, name) VALUES (%s, %s)"
            ) 
            try:
                conn.execute(insert_name_map_sql, [fei, name.lower()])
            except Exception as e:
                print(e)

    conn.commit()
    conn.close()

    return fei
