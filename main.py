from dxdata import DXDataset
# from model import DXVAE
from model import DXVAE
import torch

# print(model.state_dict().keys(), '\n')
# optim = torch.optim.SGD(model.parameters(), lr=0.01)
# print(optim.state_dict())

# checkpoint = {
#     'epoch': 90,
#     'model_state': model.state_dict(),
#     'optim_state': optim.state_dict()
# }
# torch.save(checkpoint, 'checkpoint')
# checkpoint = torch.load('checkpoint')
# model.load_state_dict(checkpoint['model_state'])
# optim.load_state_dict(checkpoint['optim_state'])

# save on cpu, load on gpu
# torch.save(model.state_dict(), PATH)
# device = torch.device('cuda')
# model.load_state_dict(torch.load(PATH, map_location='cuda:0'))  # if using a new device
# model.to(device)

# for p in model.parameters():
# print(p.size())

# model.eval()

def print_data(G):
    for g in G:
        # print(g.ndata['params'])
        print(g.ndata['X'])
        # print(g.ndata['params_1hot'].size())
        print(g.edges())

# from models import DVAE
# def test_DVAE():
#     g = igraph.Graph(directed=True)
#     g.add_vertices(5)
#     g.vs['type'] = [0,1,2,3,2]
#     g.add_edges([(2, 3), (1, 3), (0, 1), (3, 4), (0, 4)])
#
#     g1 = igraph.Graph(directed=True)
#     g1.add_vertices(5)
#     g1.vs['type'] = [0,3,2,2,1]
#     g1.add_edges([(2, 3), (1, 3), (0, 1), (3, 4), (0, 4)])
#
#     m = DVAE(max_n=5, nvt=4, START_TYPE=0, END_TYPE=3, hs=6, nz=2)
#     print(m.forward([g, g1]))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DXDataset(raw_dir='DX_data')
    G = [g.to(device) for g in DXDataset(raw_dir='DX_data')[0]]

    checkpoint = 'test_new_0920'
    # model = DXVAE(checkpoint=checkpoint)
    model = DXVAE()
    model.to(device)
    model.train(G, 500, checkpoint=checkpoint)

    # g = G[13:14]
    # _, l_x, l_e, kld = model.forward(g)
    # print(f'lp: {lp:.4f}\tle: {le:.4f}\tkld: {kld:.4f}')

    # print(model.encode(g).loc)

    # G_gen = model.generate(1)
    # print_data(G_gen)

    # print_data(G)
    # g_re = model.encode_decode(g)  # try stochastic=True
    # print_data(g_re)

    # to_syx(G_re)


# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
#  + stochastic (logit/prob)
#  RNN for param decoding + DAGNN for encoding
#  decode solidify params
#  levels for edge weights
#  param layer separation
#  training plot
#  try more layers
#  try one decode unit for one frontier (needs dgl.reorder_graph)
#  all integer params!
