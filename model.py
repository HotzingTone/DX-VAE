import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import dgl


class DXVAE(nn.Module):
    def __init__(self, n_nodes=7, n_params=21, size_X=27, size_X0=23, size_H=1024, size_Z=256, checkpoint=None):
        super().__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_nodes = n_nodes  # total number of nodes
        self.n_params = n_params  # number of node parameters
        self.size_X = size_X  # size of training data for a node
        self.size_Z = size_Z  # size of latent z
        self.zero_X0 = torch.zeros(size_X0).to(self.device)  # padding X for node_0
        self.zero_H = torch.zeros(size_H).to(self.device)  # padding hidden state for propagation
        self.hidden = None  # hidden state container
        self.p_dist = Normal(0., 1.)  # prior distribution

        # encoder
        self.combin_encode = nn.GRUCell(size_X, size_H)  # encoder combiner
        self.loop_encode = nn.GRUCell(size_X, size_H)  # encoder self-looper
        self.root_encode = nn.GRUCell(size_X0, size_H)  # encode to the root
        self.h_to_mu = nn.Linear(size_H, size_Z)  # latent mean
        self.h_to_std = nn.Sequential(
            nn.Linear(size_H, size_Z),
            nn.Softplus())  # softplus to ensure positive standard deviation

        # decoder
        self.combin_decode = nn.GRUCell(size_X, size_H)  # decoder combiner
        self.loop_decode = nn.GRUCell(size_X, size_H)  # decoder self-looper
        self.root_decode = nn.GRUCell(size_X0, size_H)  # decode from the root
        self.z_to_h = nn.Sequential(
            nn.Linear(size_Z, size_H),
            nn.Tanh()
        )  # latent z to initial hidden state H_init
        self.h_to_x0 = nn.Sequential(
            nn.Linear(size_H, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, size_X0 + 32),  # plus 32 algorithm classes
        )  # node hidden state to parameters of node-0
        self.h_to_x = nn.Sequential(
            nn.Linear(size_H, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, size_X),
        )  # node hidden state to parameters
        self.h_to_edge_self = nn.Sequential(
            nn.Linear(size_H, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, 1),
        )
        self.h_to_edge = nn.Sequential(
            nn.Linear(size_H * 2, size_H * 4),
            nn.ReLU(),
            nn.Linear(size_H * 4, 2),
        )  # scores for edge vj->vi & vi->vj given (Hvi, Hvj)

        # gated sum for propagation
        self.gate = nn.Sequential(
            nn.Linear(size_H * 2, size_H),
            nn.Sigmoid()
        )
        self.mapper = nn.Sequential(
            nn.Linear(size_H * 2, size_H, bias=False),
        )  # disable bias to ensure padded zeros also mapped to zeros

        # loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        # optionally load checkpoint
        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def _get_hidden(self, v):
        Hv = torch.stack([h[v] for h in self.hidden])
        return Hv

    def _q_lin(self, x, scale):
        p = x * scale
        p.round_().clamp_(0, scale)
        x = p / scale
        return x, p

    def _q_log(self, x, scale):
        log_scale = (torch.tensor(scale + 1)).log()
        p = (x * log_scale).exp() - 1
        p.round_().clamp_(0, scale)
        x = (p + 1).log() / log_scale
        return x, p

    def _q_bool(self, x):
        p = x.sigmoid().round()
        return p, p

    def _q_prob(self, x, n_classes):
        p = torch.argmax(x, dim=1).long()
        x = F.one_hot(p, n_classes)
        return x, p

    def _reg_x0(self, X0_plus):
        p0 = torch.zeros(X0_plus.size()[0], self.n_params)
        X0 = torch.zeros(X0_plus.size()[0], self.size_X)
        X0[:, :23] = X0_plus[:, :23]

        X0[:, :8], p0[:, :8] = self._q_lin(X0[:, :8], 99)  # p_env
        X0[:, 8], p0[:, 8] = self._q_lin(X0[:, 8], 48)  # tsp
        X0[:, 9:13], p0[:, 9:13] = self._q_lin(X0[:, 9:13], 99)  # lfs, lfd, lpmd, lamd
        X0[:, 13:15], p0[:, 13:15] = self._q_lin(X0[:, 13:15], 7)  # fb, lpms

        X0[:, 15:17], p0[:, 15:17] = self._q_bool(X0[:, 15:17])  # oks, lks

        X0[:, 17:23], p0[:, 17] = self._q_prob(X0[:, 17:23], 6)  # lfw

        _, p0[:, 18] = self._q_prob(X0_plus[:, 23:], 32)  # alg

        return X0, p0

    def _reg_xi(self, Xi, n):
        pi = torch.zeros(n, self.n_params)

        Xi[:, :9], pi[:, :9] = self._q_lin(Xi[:, :9], 99)  # lev, env
        Xi[:, 11], pi[:, 11] = self._q_lin(Xi[:, 11], 14)  # det
        Xi[:, 12:15], pi[:, 12:15] = self._q_lin(Xi[:, 12:15], 99)  # bp, ld, rd
        Xi[:, 15], pi[:, 15] = self._q_lin(Xi[:, 15], 3)  # ams
        Xi[:, 16:18], pi[:, 16:18] = self._q_lin(Xi[:, 16:18], 7)  # kvs, rs

        Xi[:, 18], pi[:, 18] = self._q_bool(Xi[:, 18])  # mode

        Xi[:, 19:23], pi[:, 19] = self._q_prob(Xi[:, 19:23], 4)  # lc
        Xi[:, 23:27], pi[:, 20] = self._q_prob(Xi[:, 23:26], 4)  # rc

        for idx, mode in enumerate(pi[:, 18]):
            if mode == 0:
                Xi[idx, 9], pi[idx, 9] = self._q_log(Xi[idx, 9], 31)  # fc
                Xi[idx, 10], pi[idx, 10] = self._q_log(Xi[idx, 10], 99)  # ff
            else:
                Xi[idx, 9], pi[idx, 9] = self._q_lin(Xi[idx, 9], 3)  # fc
                Xi[idx, 10], pi[idx, 10] = self._q_lin(Xi[idx, 10], 99)  # fc

        return Xi, pi

    def _propagate(self, G, v, H_in=None, encode=False):
        if encode:
            neighbors = range(v + 1, self.n_nodes)
            rooter = self.root_encode
            combiner = self.combin_encode
            looper = self.loop_encode
        else:
            neighbors = range(v - 1, -1, -1)
            rooter = self.root_decode
            combiner = self.combin_decode
            looper = self.loop_decode
        # compute H_in
        if H_in is None:
            H_forth = torch.stack(
                [torch.stack(
                    [self.hidden[idx][x]
                     if x in g.predecessors(v)
                     else self.zero_H
                     for x in neighbors]
                ) for idx, g in enumerate(G)])
            H_back = torch.stack(
                [torch.stack(
                    [self.hidden[idx][x]
                     if x in g.successors(v)
                     else self.zero_H
                     for x in neighbors]
                ) for idx, g in enumerate(G)])
            # (size_batch, n_node-v-1, size_H) for encoding
            # (size_batch, v, size_H) for decoding
            H_in = torch.cat([H_forth, H_back], 2)
            H_in = (self.gate(H_in) * self.mapper(H_in)).sum(1)  # (size_batch, size_H)
        # combine with X
        X = torch.stack([g.ndata['X'][v] for g in G]).to(self.device)
        if v == 0:
            X0 = X[:, :23]
            Hv = rooter(X0, H_in)  # (size_batch, size_H)
        else:
            X_loop = torch.zeros_like(X)
            for idx, g in enumerate(G):
                if v in g.successors(v):  # if self-loop
                    X_loop[idx] = X[idx]  # copy x to x_loop
            Hv = combiner(X, H_in)  # (size_batch, size_H)
            Hv = looper(X_loop, Hv)  # (size_batch, size_H)
        # store Hv into G
        for idx in range(len(G)):
            self.hidden[idx][v] = Hv[idx]

        return Hv

    def encode(self, G):
        self.hidden = [[None] * self.n_nodes for _ in range(len(G))]  # placeholder
        # from the leaf
        H_init = self.zero_H.repeat(len(G), 1)
        self._propagate(G, self.n_nodes - 1, H_init, encode=True)
        # down stream
        for v in range(self.n_nodes - 2, -1, -1):
            self._propagate(G, v, encode=True)
        # to the root
        Hg = self._get_hidden(0)
        mu, std = self.h_to_mu(Hg), self.h_to_std(Hg)
        q_dist = Normal(mu, std)
        return q_dist

    def decode(self, z):
        # make new graphs with node_0 having X0 & H0
        H_init = self.z_to_h(z)
        X0_plus = self.h_to_x0(H_init)
        X0, p0 = self._reg_x0(X0_plus)
        G = [dgl.graph(([], [])).to(self.device) for _ in range(len(z))]
        for idx, g in enumerate(G):
            g.add_nodes(1, {'X': X0[idx].unsqueeze(0),
                            'params': p0[idx].unsqueeze(0)})  # add X0 and params-0
        self._propagate(G, 0, H_init)  # compute H0 given X0 & H_init

        # generate nodes/edges 1-by-1 from node_1 upwards
        for vi in range(1, self.n_nodes):
            # generate parameters Xi from Hi-1
            Hg = self._get_hidden(vi - 1)
            Xi = self.h_to_x(Hg)
            Xi, pi = self._reg_xi(Xi, z.size()[0])
            for idx, g in enumerate(G):
                g.add_nodes(1, {'X': Xi[idx].unsqueeze(0),
                                'params': pi[idx].unsqueeze(0)})
            Hi = self._propagate(G, vi)
            # generate self-loop-edge
            Ei_self = self.h_to_edge_self(Hi).sigmoid() > 0.5  # (size_batch, 1) boolean
            for idx, g in enumerate(G):
                if Ei_self[idx]:
                    g.add_edges(vi, vi)
            Hi = self._propagate(G, vi)
            # generate in/out-edges
            for vj in range(vi - 1, -1, -1):
                Hj = self._get_hidden(vj)
                # generate in/out edge probabilities
                e_ij = self.h_to_edge(torch.cat([Hi, Hj], -1)).sigmoid() > 0.5  # (size_batch, 2) in/out boolean
                for idx, g in enumerate(G):
                    if e_ij[idx, 0]:  # in-edge, j->i
                        g.add_edges(vj, vi)
                    if e_ij[idx, 1]:  # out-edge, i->j
                        g.add_edges(vi, vj)
                Hi = self._propagate(G, vi)

        return G

    def encode_decode(self, G_true, stochastic=False):
        q_dist = self.encode(G_true)
        if stochastic:
            z = q_dist.sample().to(self.device)
        else:
            z = q_dist.loc
        G = self.decode(z)
        return G

    def generate(self, n):
        self.hidden = [[None] * self.n_nodes for _ in range(n)]  # placeholder
        sample = self.p_dist.sample((n, self.size_Z)).to(self.device)
        G = self.decode(sample)
        return G

    def loss(self, q_dist, G_true, w_env=2, w_frq=5, w_kld=0.01):
        # get true X and adjacency matrix
        X_true = torch.stack(
            [g_true.ndata['X']
             for g_true in G_true])  # (size_batch, n_nodes, size_X)
        params_true = torch.stack(
            [g_true.ndata['params']
             for g_true in G_true])  # (size_batch, n_nodes, 21)
        adj_true = torch.stack(
            [g_true.adj().to_dense().to(self.device)
             for g_true in G_true])  # (size_batch, n_nodes, n_nodes)

        # compute H_init, X0 from latent
        if self.training:
            z = q_dist.rsample()
        else:
            z = q_dist.loc

        H_init = self.z_to_h(z)
        X0 = self.h_to_x0(H_init)

        # teacher forcing X0_true to new graphs and compute H0
        X0_true = X_true[:, 0, :]
        p0_true = params_true[:, 0, :].long()
        G = [dgl.graph(([], [])).to(self.device) for _ in range(len(z))]
        for idx, g in enumerate(G):
            g.add_nodes(1, {'X': X0_true[idx].unsqueeze(0)})  # teacher forcing X0_true
        self._propagate(G, 0, H_init)  # compute H0 given X0_true & H_init

        # compute loss_X0
        loss_X0 = 0
        loss_Xi = 0
        loss_E = 0
        loss_X0 += self.mse(X0[:, :8] * w_env, X0_true[:, :8] * w_env).mean(0).sum()  # add weight to env
        loss_X0 += self.mse(X0[:, 8] * w_frq, X0_true[:, 8] * w_frq).mean(0).sum()  # add weight to tsp
        loss_X0 += self.mse(X0[:, 9:15], X0_true[:, 9:15]).mean(0).sum()  # see parse_global() in dxdata.py
        loss_X0 += self.bce(X0[:, 15:17], X0_true[:, 15:17]).mean(0).sum()  # oks,lks - osc & lfo key sync
        loss_X0 += self.ce(X0[:, 17:23], p0_true[:, 17]).mean()  # lfw - 6 lfo waveforms
        loss_X0 += self.ce(X0[:, 23:], p0_true[:, 18]).mean()  # alg - 32 algorithms

        # teacher forcing nodes/edges 1-by-1 and compute/accumulate losses
        for vi in range(1, self.n_nodes):
            # generate parameters Xi from Hi-1
            Hg = self._get_hidden(vi - 1)
            Xi = self.h_to_x(Hg)
            # teacher forcing true parameters Xi_true to compute Hi
            Xi_true = X_true[:, vi, :]
            pi_true = params_true[:, vi, :].long()
            for idx, g in enumerate(G):
                g.add_nodes(1, {'X': Xi_true[idx].unsqueeze(0)})
            Hi = self._propagate(G, vi)

            # compute parameter loss
            loss_Xi += self.mse(Xi[:, :9] * w_env, Xi_true[:, :9] * w_env).mean(0).sum()  # add weight to lev, env
            loss_Xi += self.mse(Xi[:, 9] * w_frq, Xi_true[:, 9] * w_frq).mean(0).sum()  # add weight to fc
            loss_Xi += self.mse(Xi[:, 10:18], Xi_true[:, 10:18]).mean(0).sum()  # see parse_op() in dxdata.py
            loss_Xi += self.bce(Xi[:, 18], Xi_true[:, 18]).mean()  # oks,lks - osc & lfo key sync
            loss_Xi += self.ce(Xi[:, 19:23], pi_true[:, 19]).mean()  # lc - 4 left curves
            loss_Xi += self.ce(Xi[:, 23:27], pi_true[:, 20]).mean()  # rc - 4 right curves

            # generate self-loop-edge probability
            Ei_self = self.h_to_edge_self(Hi)
            # teacher forcing Ei_self_true to update Hi
            Ei_self_true = adj_true[:, vi, vi].unsqueeze(1)  # (size_batch, 1)
            for idx, g in enumerate(G):
                if Ei_self_true[idx]:
                    g.add_edges(vi, vi)
            Hi = self._propagate(G, vi)
            # compute self-loop-edge loss
            loss_E += self.bce(Ei_self, Ei_self_true).mean()

            # consider in/out-edge probabilities
            Ei = []
            adj_in_true = adj_true[:, :vi, vi].unsqueeze(2)  # (size_batch, vi, 1)
            adj_out_true = adj_true[:, vi, :vi].unsqueeze(2)  # (size_batch, vi, 1)
            Ei_true = torch.cat([adj_in_true, adj_out_true], 2)  # (size_batch, vi, 2) in/out adj
            # count edges from edge_i_i-1 ... edge_i_0
            for vj in range(vi - 1, -1, -1):
                Hj = self._get_hidden(vj)
                # generate in/out edge probabilities
                e_ij = self.h_to_edge(torch.cat([Hi, Hj], -1)).unsqueeze(1)  # (size_batch, 1, 2) in/out prob
                Ei.append(e_ij)
                # teacher forcing true edges and update Hi
                for idx, g in enumerate(G):
                    if adj_in_true[idx, vj]:
                        g.add_edges(vj, vi)
                    if adj_out_true[idx, vj]:
                        g.add_edges(vi, vj)
                Hi = self._propagate(G, vi)
            # resorted list as edge_i_0 ... edge_i_i-1
            Ei.reverse()
            Ei = torch.cat(Ei, 1)  # (size_batch, vi, 2) in/out prob
            # compute out/in-edge losses
            loss_E += self.bce(Ei, Ei_true).mean(0).sum()

        kld = kl_divergence(self.p_dist, q_dist).mean(0).sum()

        return loss_X0 + loss_Xi + loss_E + kld * w_kld, loss_X0, loss_Xi, loss_E, kld * w_kld

    def forward(self, G_true, w_env=2, w_frq=5, w_kld=0.01):
        q_dist = self.encode(G_true)
        loss, lx0, lxi, le, kld = self.loss(q_dist, G_true, w_env, w_frq, w_kld)
        return loss, lx0, lxi, le, kld

    def train(self, G_true, epochs, size_batch=32, lr=0.001, checkpoint=None, w_env=2, w_frq=5, w_kld=0.01):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        n_samples = len(G_true)
        n_iters = int(n_samples / size_batch)
        for epoch in range(epochs + 1):
            print(f'Epoch: {epoch}')
            random.shuffle(G_true)
            for i in range(n_iters):
                G_batch = G_true[i*size_batch:(i+1)*size_batch]
                optimizer.zero_grad()
                loss, lx0, lxi, le, kld = self.forward(G_batch, w_env, w_frq, w_kld)
                loss.backward()
                optimizer.step()
                print(f'batch: {i}\tloss: {loss:.4f}\tx0: {lx0:.4f}\txi: {lxi:.4f}\te: {le:.4f}\tkld: {kld:.4f}')
            torch.save(self.state_dict(), checkpoint)
            print(f'\nCheckpoint [{checkpoint}] saved\n')

        print('Finished Training')
