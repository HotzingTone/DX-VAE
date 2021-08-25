from dxdata import DXDataset
from model import DXVAE


if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    G = dataset[0]  # dataset: (graphs, labels)
    # g = G[0]
    # print(g.edges())

    model = DXVAE()
    model.train(G, 2)

    # G_re = model.encode_decode(G)  # try stochastic=True
    # G_gen = model.generate(1)


# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
#  check in-place tensor operations
#  torch probability replacement - KLD / log-likelihood / + stochastic
#  loss func to use mean
