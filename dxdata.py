import os
from pathlib import Path
import mido
import torch
import torch.nn.functional as F
import dgl

'''
byte             bit #
 #     6   5   4   3   2   1   0   param A       range  param B       range
----  --- --- --- --- --- --- ---  ------------  -----  ------------  -----
  0                R1              OP6 EG R1      0-99
  1                R2              OP6 EG R2      0-99
  2                R3              OP6 EG R3      0-99
  3                R4              OP6 EG R4      0-99
  4                L1              OP6 EG L1      0-99
  5                L2              OP6 EG L2      0-99
  6                L3              OP6 EG L3      0-99
  7                L4              OP6 EG L4      0-99
  8                BP              LEV SCL BRK PT 0-99
  9                LD              SCL LEFT DEPTH 0-99
 10                RD              SCL RGHT DEPTH 0-99
 11    0   0   0 |  RC   |   LC  | SCL RGHT CURVE 0-3   SCL LEFT CURVE 0-3
 12  |      DET      |     RS    | OSC DETUNE     0-14  OSC RATE SCALE 0-7
 13    0   0 |    KVS    |  AMS  | KEY VEL SENS   0-7   AMP MOD SENS   0-3
 14                OL              OP6 OUTPUT LEV 0-99
 15    0 |         FC        | M | FREQ COARSE    0-31  OSC MODE       0-1
 16                FF              FREQ FINE      0-99
 17 \
  |  > these 17 bytes for OSC 5
 33 /
 34 \
  |  > these 17 bytes for OSC 4
 50 /
 51 \
  |  > these 17 bytes for OSC 3
 67 /
 68 \
  |  > these 17 bytes for OSC 2
 84 /
 85 \
  |  > these 17 bytes for OSC 1
101 /

byte             bit #
 #     6   5   4   3   2   1   0   param A       range  param B       range
----  --- --- --- --- --- --- ---  ------------  -----  ------------  -----
102               PR1              PITCH EG R1   0-99
103               PR2              PITCH EG R2   0-99
104               PR3              PITCH EG R3   0-99
105               PR4              PITCH EG R4   0-99
106               PL1              PITCH EG L1   0-99
107               PL2              PITCH EG L2   0-99
108               PL3              PITCH EG L3   0-99
109               PL4              PITCH EG L4   0-99
110    0   0 |        ALG        | ALGORITHM     0-31
111    0   0   0 |OKS|    FB     | OSC KEY SYNC  0-1    FEEDBACK      0-7
112               LFS              LFO SPEED     0-99
113               LFD              LFO DELAY     0-99
114               LPMD             LF PT MOD DEP 0-99
115               LAMD             LF AM MOD DEP 0-99
116  |  LPMS     |    LFW    |LKS| LF PT MOD SNS 0-7   WAVE 0-5,  SYNC 0-1
117              TRNSP             TRANSPOSE     0-48
118          NAME CHAR 1           VOICE NAME 1  ASCII
119          NAME CHAR 2           VOICE NAME 2  ASCII
120          NAME CHAR 3           VOICE NAME 3  ASCII
121          NAME CHAR 4           VOICE NAME 4  ASCII
122          NAME CHAR 5           VOICE NAME 5  ASCII
123          NAME CHAR 6           VOICE NAME 6  ASCII
124          NAME CHAR 7           VOICE NAME 7  ASCII
125          NAME CHAR 8           VOICE NAME 8  ASCII
126          NAME CHAR 9           VOICE NAME 9  ASCII
127          NAME CHAR 10          VOICE NAME 10 ASCII
'''


class DXDataset(dgl.data.DGLDataset):
    """ The basic DGL dataset for creating graph datasets.
        This class defines a basic template class for DGL Dataset.
        The following steps will be executed automatically:

          1. Check whether there is a dataset cache on disk
             (already processed and stored on the disk) by
             invoking ``has_cache()``. If true, goto 5.
          2. Call ``download()`` to download the data.
          3. Call ``process()`` to process the data.
          4. Call ``save()`` to save the processed dataset on disk and goto 6.
          5. Call ``load()`` to load the processed dataset from disk.
          6. Done.

        Users can overwite these functions with their
        own data processing logic.

        Parameters
        ----------
        name : str
            Name of the dataset
        url : str
            Url to download the raw dataset
        raw_dir : str
            Specifying the directory that will store the
            downloaded data or the directory that
            already stores the input data.
            Default: ~/.dgl/
        save_dir : str
            Directory to save the processed dataset.
            Default: same as raw_dir
        hash_key : tuple
            A tuple of values as the input for the hash function.
            Users can distinguish instances (and their caches on the disk)
            from the same dataset class by comparing the hash values.
            Default: (), the corresponding hash value is ``'f9065fa7'``.
        force_reload : bool
            Whether to reload the dataset. Default: False
        verbose : bool
            Whether to print out progress information

        Attributes
        ----------
        url : str
            The URL to download the dataset
        name : str
            The dataset name
        raw_dir : str
            Raw file directory contains the input data folder
        raw_path : str
            Directory contains the input data files.
            Default : ``os.path.join(self.raw_dir, self.name)``
        save_dir : str
            Directory to save the processed dataset
        save_path : str
            File path to save the processed dataset
        verbose : bool
            Whether to print information
        hash : str
            Hash value for the dataset and the setting.
        """

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
        def parse_op(idx):
            i = (6 - idx) * 17

            env = torch.clamp(pz[i:i + 8], 0, 99)  # r1...r4,l1...l4

            bp = torch.clamp(pz[i + 8], 0, 99)  # breakpoint
            ld = torch.clamp(pz[i + 9], 0, 99)  # left depth
            rd = torch.clamp(pz[i + 10], 0, 99)  # right depth

            rc = torch.floor(pz[i + 11] / 4) % 4  # right 4 curves
            lc = pz[i + 11] % 4  # left 4 curves

            det = torch.clamp(torch.floor(pz[i + 12] / 8), 0, 14)  # detune
            rs = pz[i + 12] % 8  # rate scale

            kvs = torch.floor(pz[i + 13] / 4) % 8  # keyboard velocity sensitivity
            ams = pz[i + 13] % 4  # amplitude modulation sensitivity

            lev = torch.clamp(pz[i + 14], 0, 99)  # output level

            fc = torch.floor(pz[i + 15] / 2) % 32  # coarse frequency
            mode = pz[i + 15] % 2  # ratio / fixed frequency mode, boolean

            ff = torch.clamp(pz[i + 16], 0, 99)  # fine frequency

            if mode == 0:
                fc_x = (fc + 1).log() / (torch.tensor(32.)).log()
                ff_x = (ff + 1).log() / (torch.tensor(100.)).log()
            else:
                fc = fc % 4
                fc_x = fc / 3
                ff_x = ff / 99

            pi = torch.cat([lev.unsqueeze(0),  # 0
                            env,  # 1...8
                            fc.unsqueeze(0),  # 9 - 31 or 3 max
                            ff.unsqueeze(0),  # 10
                            det.unsqueeze(0),  # 11 - 14 max
                            bp.unsqueeze(0),  # 12
                            ld.unsqueeze(0),  # 13
                            rd.unsqueeze(0),  # 14
                            ams.unsqueeze(0),  # 15 - 3 max
                            kvs.unsqueeze(0),  # 16 - 7 max
                            rs.unsqueeze(0),  # 17 - 7 max

                            mode.unsqueeze(0),  # 18 - boolean

                            lc.unsqueeze(0),  # 19 - 4 curves
                            rc.unsqueeze(0),  # 20 - 4 curves
                            ])

            Xi = torch.cat([lev.unsqueeze(0) / 99,  # 0
                            env / 99,  # 1...8
                            fc_x.unsqueeze(0),  # 9
                            ff_x.unsqueeze(0),  # 10
                            det.unsqueeze(0) / 14,  # 11 - 14 max
                            bp.unsqueeze(0) / 99,  # 12
                            ld.unsqueeze(0) / 99,  # 13
                            rd.unsqueeze(0) / 99,  # 14
                            ams.unsqueeze(0) / 3,  # 15 - 3 max
                            kvs.unsqueeze(0) / 7,  # 16 - 7 max
                            rs.unsqueeze(0) / 7,  # 17 - 7 max

                            mode.unsqueeze(0),  # 18 - boolean

                            F.one_hot(lc.long(), 4),  # 19...22 - 4 curves
                            F.one_hot(rc.long(), 4),  # 23...26 - 4 curves
                            ])

            return pi, Xi

        def parse_global():
            p_env = torch.clamp(pz[102:110], 0, 99)  # r1...r4,l1...l4
            alg = pz[110] % 32  # 32 classes

            oks = torch.floor(pz[111] / 8) % 2  # osc keyboard sync, boolean
            fb = pz[111] % 8  # feedback

            lfs = torch.clamp(pz[112], 0, 99)  # lfo speed
            lfd = torch.clamp(pz[113], 0, 99)  # lfo delay
            lpmd = torch.clamp(pz[114], 0, 99)  # lfo pitch mod depth
            lamd = torch.clamp(pz[115], 0, 99)  # lfo amplitude mod depth

            lpms = torch.floor(pz[116] / 16)  # lfo pitch mod sensitivity
            lfw = torch.clamp(torch.floor(pz[116] / 2) % 8, 0, 5)  # lfo 5 waveforms
            lks = pz[116] % 2  # lfo keyboard sync, boolean

            tsp = torch.clamp(pz[117], 0, 48)  # transpose

            p0 = torch.cat([p_env,  # 0...7
                            tsp.unsqueeze(0),  # 8 - 48 max
                            lfs.unsqueeze(0),  # 9
                            lfd.unsqueeze(0),  # 10
                            lpmd.unsqueeze(0),  # 11
                            lamd.unsqueeze(0),  # 12
                            fb.unsqueeze(0),  # 13 - 7 max
                            lpms.unsqueeze(0),  # 14 - 7 max

                            oks.unsqueeze(0),  # 15 - boolean
                            lks.unsqueeze(0),  # 16 - boolean

                            lfw.unsqueeze(0),  # 17 - 6 waveforms

                            alg.unsqueeze(0),  # 18 - 32 classes

                            torch.zeros(2),  # 19...20, padding
                            ])

            X0 = torch.cat([p_env / 99,  # 0...7
                            tsp.unsqueeze(0) / 48,  # 8 - 48 max
                            lfs.unsqueeze(0) / 99,  # 9
                            lfd.unsqueeze(0) / 99,  # 10
                            lpmd.unsqueeze(0) / 99,  # 11
                            lamd.unsqueeze(0) / 99,  # 12
                            fb.unsqueeze(0) / 7,  # 13 - 7 max
                            lpms.unsqueeze(0) / 7,  # 14 - 7 max

                            oks.unsqueeze(0),  # 15 - boolean
                            lks.unsqueeze(0),  # 16 - boolean

                            F.one_hot(lfw.long(), 6),  # 17...22 - 6 waveforms

                            torch.zeros(4),  # 23...26 - padding
                            ])

            return p0, X0

        params_6_ops = torch.stack([parse_op(idx)[0] for idx in range(1, 7)])  # [6_operators, 21_params]
        params_global = parse_global()[0].unsqueeze(0)  # [1, 21]

        X_6_ops = torch.stack([parse_op(idx)[1] for idx in range(1, 7)])  # [6_operators, 28]
        X_global = parse_global()[1].unsqueeze(0)  # [1, 28]

        g = dgl.graph(self.DX_ALGO[pz[110].item()])  # make graph from edges
        g.ndata['params'] = torch.cat([params_global, params_6_ops])  # data for viewing
        g.ndata['X'] = torch.cat([X_global, X_6_ops])  # data for training

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


def graph_to_syx(G, file='gen_patch.syx'):
    data_name = [68, 88, 45, 86, 65, 69, 46, 46, 46, 46]
    data_head = [67, 0, 9, 32, 0]
    data_tail = [88]
    data_32pz = []

    for i, g in enumerate(G):
        pg = g.ndata['params'].int().tolist()
        data_pz = []
        for idx in range(6, 0, -1):
            pi = pg[idx]

            lev = pi[0]
            env = pi[1:9]
            fc = pi[9]
            ff = pi[10]
            det = pi[11]
            bp = pi[12]
            ld = pi[13]
            rd = pi[14]
            ams = pi[15]
            kvs = pi[16]
            rs = pi[17]
            mode = pi[18]
            lc = pi[19]
            rc = pi[20]

            data_op = env + [bp] + [ld] + [rd] + [rc * 4 + lc] + [det * 8 + rs] \
                      + [kvs * 4 + ams] + [lev] + [fc * 2 + mode] + [ff]
            data_pz.extend(data_op)

        p0 = pg[0]

        p_env = p0[0:8]
        tsp = p0[8]
        lfs = p0[9]
        lfd = p0[10]
        lpmd = p0[11]
        lamd = p0[12]
        fb = p0[13]
        lpms = p0[14]
        oks = p0[15]
        lks = p0[16]
        lfw = p0[17]
        alg = p0[18]

        data_global = p_env + [alg] + [oks * 8 + fb] + [lfs] + [lfd] \
                      + [lpmd] + [lamd] + [lpms * 16 + lfw * 2 + lks] + [tsp] \
                      + data_name

        data_pz.extend(data_global)
        data_32pz.extend(data_pz)

    data = data_head + data_32pz + data_tail
    msg = mido.Message('sysex', data=data)

    mido.write_syx_file(file, [msg])


if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    G = dataset[0]
