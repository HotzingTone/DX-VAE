import os
from pathlib import Path
import mido
import torch
import torch.nn.functional as F
import dgl


class DXDataset(dgl.data.DGLDataset):
    def __init__(self, raw_dir=None, save_dir=None):
        self.DX_ALGO = {0: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 0, 3, 4, 5, 6]),
                        1: ([1, 2, 2, 3, 4, 5, 6], [0, 1, 2, 0, 3, 4, 5]),
                        2: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 2, 0, 4, 5, 6]),
                        3: ([1, 2, 3, 4, 4, 5, 6], [0, 1, 2, 0, 6, 4, 5]),
                        4: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 0, 3, 0, 5, 6]),
                        5: ([1, 2, 3, 4, 5, 5, 6], [0, 1, 0, 3, 0, 6, 5]),
                        6: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 0, 3, 3, 5, 6]),
                        7: ([1, 2, 3, 4, 4, 5, 6], [0, 1, 0, 3, 4, 3, 5]),
                        8: ([1, 2, 2, 3, 4, 5, 6], [0, 1, 2, 0, 3, 3, 5]),
                        9: ([1, 2, 3, 3, 4, 5, 6], [0, 1, 2, 3, 0, 4, 4]),
                        10: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 2, 0, 4, 4, 6]),
                        11: ([1, 2, 2, 3, 4, 5, 6], [0, 1, 2, 0, 3, 3, 3]),
                        12: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 0, 3, 3, 3, 6]),
                        13: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 0, 3, 4, 4, 6]),
                        14: ([1, 2, 2, 3, 4, 5, 6], [0, 1, 2, 0, 3, 4, 4]),
                        15: ([1, 2, 3, 4, 5, 6, 6], [0, 1, 1, 3, 1, 5, 6]),
                        16: ([1, 2, 2, 3, 4, 5, 6], [0, 1, 2, 1, 3, 1, 5]),
                        17: ([1, 2, 3, 3, 4, 5, 6], [0, 1, 1, 3, 1, 4, 5]),
                        18: ([1, 2, 3, 4, 5, 6, 6, 6], [0, 1, 2, 0, 0, 4, 5, 6]),
                        19: ([1, 2, 3, 3, 3, 4, 5, 6], [0, 0, 1, 2, 3, 0, 4, 4]),
                        20: ([1, 2, 3, 3, 3, 4, 5, 6, 6], [0, 0, 1, 2, 3, 0, 0, 4, 5]),
                        21: ([1, 2, 3, 4, 5, 6, 6, 6, 6], [0, 1, 0, 0, 0, 3, 4, 5, 6]),
                        22: ([1, 2, 3, 4, 5, 6, 6, 6], [0, 0, 2, 0, 0, 4, 5, 6]),
                        23: ([1, 2, 3, 4, 5, 6, 6, 6, 6], [0, 0, 0, 0, 0, 3, 4, 5, 6]),
                        24: ([1, 2, 3, 4, 5, 6, 6, 6], [0, 0, 0, 0, 0, 4, 5, 6]),
                        25: ([1, 2, 4, 3, 5, 6, 6], [0, 0, 0, 2, 4, 4, 6]),
                        26: ([1, 2, 3, 3, 4, 5, 6], [0, 0, 2, 3, 0, 4, 4]),
                        27: ([1, 2, 3, 4, 5, 5, 6], [0, 1, 0, 3, 4, 5, 0]),
                        28: ([1, 2, 3, 4, 5, 6, 6], [0, 0, 0, 3, 0, 5, 6]),
                        29: ([1, 2, 3, 4, 5, 5, 6], [0, 0, 0, 3, 4, 5, 0]),
                        30: ([1, 2, 3, 4, 5, 6, 6], [0, 0, 0, 0, 0, 5, 6]),
                        31: ([1, 2, 3, 4, 5, 6, 6], [0, 0, 0, 0, 0, 0, 6])}
        super().__init__(name='DXDataset.bin', raw_dir=raw_dir, save_dir=save_dir)

    def _make_graph(self, pz):
        g = dgl.graph(self.DX_ALGO[pz[110].item()])  # edges

        def parse_op(idx):
            i = (6 - idx) * 17
            envelop = pz[i:i + 8] * 0.01  # 0...99 scale to 0.00...0.99
            gain = pz[i + 14] * 0.01  # 0.00...0.99
            mode = pz[i + 15] % 2  # boolean
            coarse = torch.floor(pz[i + 15] * 0.5)  # 0...31
            fine = pz[i + 16]  # 0...99
            tune = torch.floor(pz[i + 12] / 8) / 15  # 0...14/15
            if mode == 0:  # ratio mode
                if coarse == 0:
                    coarse = torch.tensor(0.5)  # ratio=0.5 when coarse=0, as DX's design
                freq_add = coarse * (1 + fine * 0.01)  # 0.5...61.69
                freq = torch.log(freq_add * 2) / torch.log(torch.tensor(128.))  # log normalization
            else:  # fixed mode
                freq = (coarse % 4 + fine * 0.01) / 4  # already log as DX's design
            op_params = torch.cat([gain.unsqueeze(0),
                                   envelop,
                                   mode.unsqueeze(0),
                                   freq.unsqueeze(0),
                                   tune.unsqueeze(0),
                                   ])
            return op_params

        pz_params = torch.stack([parse_op(idx) for idx in range(1, 7)])  # [6_operators, 12_params]
        g.ndata['params'] = torch.cat([torch.zeros(1, 12), pz_params])  # features zero-padded for node_0
        return g

    def _read_syx(self, file):
        msg = mido.read_syx_file(file)[0]
        data = torch.tensor(msg.data)
        data = data[5:-1].reshape(32, -1)
        return data  # [32_pzs, 128_params]

    def process(self):
        folder = Path(self._raw_dir).rglob('*.syx')
        raw = torch.cat([self._read_syx(file) for file in folder])  # [n_files * 32_pzs, 128_params]
        self.graphs = [self._make_graph(pz) for pz in raw]

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        dgl.save_graphs(self.save_path, self.graphs)

    def load(self):
        self.graphs = dgl.load_graphs(self.save_path)

    def has_cache(self):
        return os.path.exists(self.save_path)


# def to_syx(G, file='gen_patch.syx'):
#     data_pz_tail = [99, 99, 99, 99,
#                     50, 50, 50, 50,
#                     0, 15, 0, 0, 0,
#                     0, 1, 24, 0, 0,
#                     0, 0, 0, 0,
#                     0, 0, 0, 0]
#     data_head = [67, 0, 9, 32, 0]
#     data_tail = [88]
#     data_32pz = []
#     for i, g in enumerate(G):
#         params_graph = g.ndata['params']
#         params_pz = []
#         for idx in range(6, 0, -1):  # todo make a clamp
#             params_node = params_graph[idx]
#
#             mode_op = params_node[0].int().tolist()
#             coarse_op = params_node[1].int().tolist()
#             fine_op = params_node[2].int().tolist()
#             tune_op = params_node[3].int().tolist()
#             env_op = params_node[4:12].int().tolist()
#             level_op = params_node[12].int().tolist()
#             params_op = env_op + [0, 0, 0, 0] + \
#                         [tune_op * 8] + [0] + \
#                         [level_op] + \
#                         [coarse_op * 2 + mode_op] + \
#                         [fine_op]
#             params_pz.extend(params_op)
#
#         data_pz = params_pz + data_pz_tail
#         data_32pz.extend(data_pz)
#
#     data = data_head + data_32pz + data_tail
#     msg = mido.Message('sysex', data=data)
#
#     mido.write_syx_file(file, [msg])


if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    G = dataset[0]
    # dataset.to_presets(g)
