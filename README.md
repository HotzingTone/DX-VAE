DX-VAE is a VAE model that learns the parameters of the Dexed FM synth using Graph Learning techniques.
It sees a Dexed patch as a Computational Graph where the FM oscillators are the interconnected nodes of the graph.

DX-VAE is not a big model, and hasn't been trained on a very large dataset for the time being.
It doesn't aim to become a full-fledged product for generating Dexed pathes,
but to test the idea of Computational Graph Learning being a method of generating DSP-based sounds.

So far, the results look fine.


To try the model:
- you need pytorch, dgl, mido installed.
- then go to [main.py] to try some testing examples.
- [model.py] is where the class DXVAE is defined.
- [dxdata.py] is to make a graph dataset from *.syx synth patch files.
- *.syx files and graph dataset are in the [DX_data] folder
- [dx_1024.chk] in the [checkpoints] folder is a checkpoint of the model shortly trained on 1024 patch as early test.


