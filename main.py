from dxdata import DXDataset

if __name__ == '__main__':
    dataset = DXDataset(raw_dir='DX_data')
    g = dataset[0][0]  # dataset: (graphs, labels)
    print(g.ndata)

# Todo:
#  consider dgl.reorder_graph
#  convert back to dx presets
