from dxdata import DXDataset
from model import DXVAE
import torch
from torch.utils.data import Dataset, DataLoader



def print_data(G):
    for idx, g in enumerate(G):
        print(f'#{idx} Params:\n', g.ndata['params'])
        print(f'#{idx} Edges:\n', g.edges(), '\n')


def train_new(G, chk='auto.chk', epochs=500, size_batch=32, lr=0.001, w_env=2, w_frq=5, w_kld=0.01):
    model = DXVAE()
    model.to(device)
    model.train(G, epochs, size_batch, lr, chk, w_env, w_frq, w_kld)


def train_on(G, chk='auto.chk', epochs=500, size_batch=32, lr=0.001, w_env=2, w_frq=5, w_kld=0.01):
    model = DXVAE(checkpoint=chk)
    model.to(device)
    model.train(G, epochs, size_batch, lr, chk, w_env, w_frq, w_kld)


def decode_test(G_en, chk='auto.chk', stochastic=False):
    model = DXVAE(checkpoint=chk)
    G_de = model.encode_decode(G_en, stochastic=stochastic)
    print('[ Encode ]')
    print_data(G_en)
    print('[ Decode ]')
    print_data(G_de)


def generate_test(n=1, chk='auto.chk'):
    model = DXVAE(checkpoint=chk)
    G_gen = model.generate(n)
    print('[ Generate ]')
    print_data(G_gen)


def forward_test(G, chk='auto.chk'):
    model = DXVAE(checkpoint=chk)
    loss, lx0, lxi, le, kld = model.forward(G)
    print(f'loss: {loss:.4f}\tx0: {lx0:.4f}\txi: {lxi:.4f}\te: {le:.4f}\tkld: {kld:.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DXDataset(raw_dir='DX_data')
    G = [g.to(device) for g in dataset[0]]
    chk = 'checkpoints/dx_1024.chk'
    # train_new(G, chk=chk, epochs=100)
    train_on(G, chk=chk, epochs=100)
    # decode_test(G[10:11], chk=chk)
    # generate_test(1, chk=chk)
    # forward_test(G[15:16], chk=chk)

    # to_syx(G_gen)

# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
#  + stochastic
#  levels for edge weights
#  training plot
