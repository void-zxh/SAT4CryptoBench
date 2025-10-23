import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import pickle
import time

from g4satbench.utils.options import add_model_options
from g4satbench.utils.logger import Logger
from g4satbench.utils.utils import set_seed
from g4satbench.utils.format_print import FormatTable
from g4satbench.data.dataloader import get_dataloader, get_dataloader_prediction
from g4satbench.models.gnn import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('cnf_file', type=str, help='Path to a single .cnf file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Validation splits')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--data_fetching', type=str, default='sequential')
    parser.add_argument('--seed', type=int, default=0)
    add_model_options(parser)

    opts = parser.parse_args()
    set_seed(opts.seed)

    opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(opts)

    model = GNN(opts)
    model.to(opts.device)
    t_process_1 = time.time()
    test_loader = get_dataloader_prediction(opts.cnf_file, opts.test_splits, opts, 'test')  
    t_process_2 = time.time()
    # print(f"process anf time: {t_process_2 - t_process_1}")

    # print('Loading model checkpoint from %s..' % opts.checkpoint)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(opts.checkpoint, weights_only=False)

    assert (not opts.device.type == 'cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(opts.device)

    model.eval()

    for data in test_loader:   
        data = data.to(opts.device)
        batch_size = data.num_graphs
        with torch.no_grad():
            if opts.task == 'satisfiability':
                t1 = time.time()
                pred = model(data)
                label = data.y
                t2 = time.time()
                # time_list.append(t2-t1)
                print(f"processing time: {t2 - t1}")
                # format_table.update(pred, label)

            elif opts.task == 'assignment':
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index

                v_pred = model(data)

                # calculate the satisfying assignments
                v_assign = (v_pred > 0.5).float()
                l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
    return v_assign


if __name__ == '__main__':
    v_assigh = main()[:32]
    # print(v_assigh)

    tensor_list = v_assigh.cpu().tolist()

    result_str = ''.join(['1' if x == 1.0 else '0' for x in tensor_list])

    print(result_str)
