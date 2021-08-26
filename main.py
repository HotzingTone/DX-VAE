from dxdata import DXDataset
from model import DXVAE
import torch


if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    G = dataset[0]  # dataset: (graphs, labels)
    # g = G[0]
    # print(g.edges())

    model = DXVAE()
    PATH = 'test_01'

    model.train(G, 100)
    torch.save(model.state_dict(), PATH)

    # model.load_state_dict(torch.load(PATH))

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

    # G_re = model.encode_decode(G)  # try stochastic=True
    G_gen = model.generate(2)
    for g in G_gen:
        print(g.ndata)
        print(g.edges())


# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
#  + stochastic
#  loss func to use mean
#  RNN for env?
#  param layer separate
