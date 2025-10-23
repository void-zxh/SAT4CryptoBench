import torch

from torch_geometric.data import Data
from g4satbench.utils.utils import literal2l_idx, literal2v_idx, literal2a_idx


class ANF(Data):
    def __init__(self,
                 l_size=None,
                 c_size=None,
                 h_size=None,
                 l_edge_index=None,
                 c_edge_index=None,
                 l1_edge_index=None,
                 l2_edge_index=None,
                 l_batch=None,
                 c_batch=None,
                 h_batch=None
                 ):
        """
        l_size: The number of literals
        c_size: The number of clauses
        l1_edge_index: The literal index in the edge list to intermediate literals
        l2_edge_index: The clause index in the edge list
        l_batch: All 1 array with the size of l_size (to map each literal node to its respective graph in a batch)
        c_batch: All 1 array with the size of c_size (to map each clause node to its respective graph in a batch)
        """
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.h_size = h_size
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        self.l1_edge_index = l1_edge_index
        self.l2_edge_index = l2_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
        self.h_batch = h_batch

    @property
    def num_edges(self):
        return self.c_edge_index.size(0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l_edge_index':
            return self.h_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'l2l_row_index' or key == 'l2l_col_index':
            return self.l_size
        elif key == 'l1_edge_index':
            return self.l_size
        elif key == 'l2_edge_index':
            return self.h_size
        elif key == 'l_batch' or key == 'h_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)

class LCG(Data):
    def __init__(self,
            l_size=None,
            c_size=None,
            l_edge_index=None,
            c_edge_index=None,
            l_batch=None,
            c_batch=None
        ):
        """
        l_size: The number of literals
        c_size: The number of clauses
        l_edge_index: The literal index in the edge list
        c_edge_index: The clause index in the edge list
        l_batch: All 1 array with the size of l_size (to map each literal node to its respective graph in a batch)
        c_batch: All 1 array with the size of c_size (to map each clause node to its respective graph in a batch)
        """
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.c_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l_edge_index':
            return self.l_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'l_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class VCG(Data):
    def __init__(self, 
            v_size=None,
            c_size=None,
            v_edge_index=None,
            c_edge_index=None,
            p_edge_index=None, 
            n_edge_index=None, 
            l_edge_index=None,
            v_batch=None,
            c_batch=None
        ):
        """
        v_size: The number of variables
        c_size: The number of clauses
        v_edge_index: The variable index in the edge list
        c_edge_index: The clause index in the edge list
        p_edge_index: The positive edge index in the edge list
        n_edge_index: The negative edge index in the edge list
        l_edge_index: The literal index in the edge list
        v_batch: All 1 array with the size of l_size (to map each variable node to its respective graph in a batch)
        c_batch: All 1 array with the size of c_size (to map each clause node to its respective graph in a batch)
        """
        super().__init__()
        self.v_size = v_size
        self.c_size = c_size
        self.v_edge_index = v_edge_index
        self.c_edge_index = c_edge_index
        self.p_edge_index = p_edge_index
        self.n_edge_index = n_edge_index
        self.l_edge_index = l_edge_index
        self.v_batch = v_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.v_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'v_edge_index':
            return self.v_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'p_edge_index' or key == 'n_edge_index':
            return self.v_edge_index.size(0)
        elif key == 'l_edge_index':
            return self.v_size * 2
        elif key == 'v_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


def construct_lcg(n_vars, clauses):
    l_edge_index_list = []
    c_edge_index_list = []
    
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            l_idx = literal2l_idx(literal)
            l_edge_index_list.append(l_idx)
            c_edge_index_list.append(c_idx)

    return LCG(
        n_vars * 2,
        len(clauses),
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars * 2, dtype=torch.long),
        torch.zeros(len(clauses), dtype=torch.long)
    )

def construct_anf(n_vars, clauses):
    print('ANF')
    l_edge_index_list = []
    c_edge_index_list = []
    l1_edge_index_list = []
    l2_edge_index_list = []
    hash_to_id = {}
    v_to_id = {}
    hash_count = 0

    hash4highorder = {}
    id_count = n_vars

    for c_idx, clause in enumerate(clauses):
        clause_sign = clause[-1]
        l_list = []
        if isinstance(clause_sign, int) and clause_sign < 0:
            # negtive clause
            for literal in clause[:-1]:
                if isinstance(literal, list):
                    hashed_l = hash(frozenset(literal))
                    if hashed_l not in hash4highorder:
                        hash4highorder[hashed_l] = id_count
                        id_count += 1
                    to_lidx = hash4highorder[hashed_l]
                    for literal_idx in literal:
                        l1_edge_index_list.append(literal_idx-1)
                        l2_edge_index_list.append(to_lidx)
                    l_list.append(to_lidx)
                else:
                    l_list.append(literal-1)
                    l1_edge_index_list.append(literal-1)
                    l2_edge_index_list.append(literal-1)
                    
            hashed_set = hash(frozenset(l_list))
            if hashed_set not in hash_to_id:
                hash_to_id[hashed_set] = hash_count
                v_to_id[hashed_set] = 1
                c_hash_id = hash_count
                hash_count += 1
            elif (v_to_id[hashed_set] & 1) == 0:
                v_to_id[hashed_set] += 1
                c_hash_id = hash_to_id[hashed_set]
            else:
                continue
            for literal in clause[:-1]:
                if isinstance(literal, list):
                    hashed_l = hash(frozenset(literal))
                    l_idx = hash4highorder[hashed_l]
                else:
                    l_idx = literal-1
                l_edge_index_list.append(l_idx)
                c_edge_index_list.append(c_hash_id*2+1)
        else:
            # positive clause
            for literal in clause:
                if isinstance(literal, list):
                    hashed_l = hash(frozenset(literal))
                    if hashed_l not in hash4highorder:
                        hash4highorder[hashed_l] = id_count
                        id_count += 1
                    to_lidx = hash4highorder[hashed_l]
                    for literal_idx in literal:
                        l1_edge_index_list.append(literal_idx-1)
                        l2_edge_index_list.append(to_lidx)
                    l_list.append(to_lidx)
                else:
                    l_list.append(literal-1)
                    l1_edge_index_list.append(literal-1)
                    l2_edge_index_list.append(literal-1)
            hashed_set = hash(frozenset(l_list))
            if hashed_set not in hash_to_id:
                hash_to_id[hashed_set] = hash_count
                v_to_id[hashed_set] = 2
                c_hash_id = hash_count
                hash_count += 1
            elif (v_to_id[hashed_set] & 2) == 0:
                v_to_id[hashed_set] += 2
                c_hash_id = hash_to_id[hashed_set]
            else:
                continue
            for literal in clause:
                if isinstance(literal, list):
                    hashed_l = hash(frozenset(literal))
                    l_idx = hash4highorder[hashed_l]
                else:
                    l_idx = literal-1
                l_edge_index_list.append(l_idx)
                c_edge_index_list.append(c_hash_id*2)

    # print(l1_edge_index_list)
    # print(l2_edge_index_list)
    # print(l_edge_index_list)
    # print(c_edge_index_list)
    # print(lll)
    return ANF(
        n_vars,
        hash_count*2,
        id_count,
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.tensor(l1_edge_index_list, dtype=torch.long),
        torch.tensor(l2_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars, dtype=torch.long),
        torch.zeros(hash_count*2, dtype=torch.long),
        torch.zeros(id_count, dtype=torch.long),
    )


def construct_vcg(n_vars, clauses):
    c_edge_index_list = []
    v_edge_index_list = []
    p_edge_index_list = []
    n_edge_index_list = []
    l_edge_index_list = []

    edge_index = 0
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            sign, v_idx = literal2v_idx(literal)
            c_edge_index_list.append(c_idx)
            v_edge_index_list.append(v_idx)
            
            if sign:
                p_edge_index_list.append(edge_index)
                l_edge_index_list.append(v_idx * 2)
            else:
                n_edge_index_list.append(edge_index)
                l_edge_index_list.append(v_idx * 2 + 1)
            
            edge_index += 1
    
    return VCG(
        n_vars,
        len(clauses),
        torch.tensor(v_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.tensor(p_edge_index_list, dtype=torch.long),
        torch.tensor(n_edge_index_list, dtype=torch.long),
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars, dtype=torch.long),
        torch.zeros(len(clauses), dtype=torch.long)
    )

if __name__ == '__main__':
    n_vars = 5
    clauses = [[1, 2, 1], [2, 1], [3, 1], [4, 1], [5, 1]]
    construct_anf(n_vars, clauses)