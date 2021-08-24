from dxdata import DXDataset
from model import DXVAE

if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    G = dataset[0]  # dataset: (graphs, labels)
    model = DXVAE()
    loss = model.forward(G)
    G_re = model.encode_decode(G)  # try stochastic=True
    G_gen = model.generate(1)

# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
#  check in-place tensor operations
#  torch probability replacement - KLD / log-likelihood / + stochastic
