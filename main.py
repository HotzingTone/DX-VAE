from dxdata import DXDataset, graph_to_syx
from model import DXVAE
import torch


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


def decode_test(G_en, chk='auto.chk', stochastic=False, print=True):
    model = DXVAE(checkpoint=chk)
    G_de = model.encode_decode(G_en, stochastic=stochastic)
    if print:
        print('[ Encode ]')
        print_data(G_en)
        print('[ Decode ]')
        print_data(G_de)
    return(G_de)


def generate_test(n=1, chk='auto.chk', print=True):
    model = DXVAE(checkpoint=chk)
    G_gen = model.generate(n)
    if print:
        print('[ Generate ]')
        print_data(G_gen)
    return(G_gen)


def forward_test(G, chk='auto.chk'):
    model = DXVAE(checkpoint=chk)
    loss, lx0, lxi, le, kld = model.forward(G)
    print(f'loss: {loss:.4f}\tx0: {lx0:.4f}\txi: {lxi:.4f}\te: {le:.4f}\tkld: {kld:.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DXDataset(raw_dir='DX_data')
    G = [g.to(device) for g in dataset[0]]
    chk = 'checkpoints/dx_1024.chk'
    # train_new(G, chk=chk, epochs=50, w_env=3, w_frq=6, w_kld=0.002)
    train_on(G, chk=chk, epochs=50, w_env=3, w_frq=6, w_kld=0.002)
    # decode_test(G[12:13], chk=chk)
    # generate_test(1, chk=chk)
    # forward_test(G[15:16], chk=chk)
    # graph_to_syx(generate_test(32, chk=chk, print=False))

# Todo:
#  consider dgl.reorder_graph
#  stochastic
#  levels for edge weights
#  training plot
