import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
import dgl


class DXVAE(nn.Module):
    def __init__(self, n_nodes=7, n_params=21, size_X=27, size_X0=23, size_H=512, size_Z=128, checkpoint=None):
        super().__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_nodes = n_nodes  # total number of nodes
        self.n_params = n_params  # number of node parameters
        self.size_X = size_X  # size of training data for a node
        self.size_H = size_H  # size of hidden state for a node
        self.size_Z = size_Z  # size of latent z
        self.zero_X = torch.zeros(size_X).to(self.device)  # padding X for node_0
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
            nn.Linear(size_H * 2, size_X0 + 32),  # plus 32 algorithm classes
        )  # node hidden state to parameters of node-0
        self.h_to_x = nn.Sequential(
            nn.Linear(size_H, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, size_X),
        )  # node hidden state to parameters
        self.h_to_edge_self = nn.Sequential(
            nn.Linear(size_H, size_H * 2),
            nn.ReLU(),
            nn.Linear(size_H * 2, 1),
            nn.Sigmoid()
        )
        self.h_to_edge = nn.Sequential(
            nn.Linear(size_H * 2, size_H * 4),
            nn.ReLU(),
            nn.Linear(size_H * 4, 2),
            nn.Sigmoid()
        )  # probabilities for edge vj->vi & vi->vj given (Hvi, Hvj)

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
        # feature processing functions
        def x0_to_p0(X0):
            p0 = torch.zeros(z.size()[0], self.n_params)

            p0[:, :12] = X0[:, :12] * 100
            torch.clamp_(torch.round_(p0[:, :12]), 0, 99)
            X0[:, :12] = p0[:, :12] / 100

            p0[:, 12:14] = X0[:, 12:14] * 8  # fb, lpms
            torch.clamp_(torch.round_(p0[:, 12:14]), 0, 7)
            X0[:, 12:14] = p0[:, 12:14] / 8

            p0[:, 14] = X0[:, 14] * 49  # tsp
            torch.clamp_(torch.round_(p0[:, 14]), 0, 48)
            X0[:, 14] = p0[:, 14] / 49

            p0[:, 15:17] = F.sigmoid(X0[:, 15:17])  # oks, lks
            torch.round_(p0[:, 15:17])
            X0[:, 15:17] = p0[:, 15:17]

            p0[:, 17] = torch.argmax(X0[:, 17:23], dim=1)  # lfw
            X0[:, 17:23] = F.one_hot(p0[:, 17], 6)

            p0[:, 18] = torch.argmax(X0[:, 23:], dim=1)  # alg

            return X0, p0

        def xi_to_pi(Xi):
            pi = torch.zeros(z.size()[0], self.n_params)

            pi[:, :13] = X0[:, :13] * 100
            torch.clamp_(torch.round_(pi[:, :13]), 0, 99)
            Xi[:, :13] = pi[:, :13] / 100

            pi[:, 13] = Xi[:, 13] * 4  # ams
            torch.clamp_(torch.round_(pi[:, 13]), 0, 3)
            Xi[:, 13] = pi[:, 13] / 4

            pi[:, 14:16] = Xi[:, 14:16] * 8  # kvs, rs
            torch.clamp_(torch.round_(pi[:, 14:16]), 0, 7)
            Xi[:, 14:16] = pi[:, 14:16] / 8

            pi[:, 16] = Xi[:, 16] * 15  # det
            torch.clamp_(torch.round_(pi[:, 16]), 0, 14)
            Xi[:, 16] = pi[:, 16] / 15

            pi[:, 17] = Xi[:, 17] * 32  # fc
            torch.clamp_(torch.round_(pi[:, 17]), 0, 31)
            Xi[:, 17] = pi[:, 17] / 32

            pi[:, 18] = F.sigmoid(Xi[:, 18])  # mode
            torch.round_(pi[:, 18])
            Xi[:, 18] = p0[:, 18]

            pi[:, 19] = torch.argmax(Xi[:, 19:23], dim=1)  # lc
            Xi[:, 19:23] = F.one_hot(pi[:, 19], 6)

            pi[:, 20] = torch.argmax(Xi[:, 23:27], dim=1)  # rc
            Xi[:, 23:27] = F.one_hot(pi[:, 20], 6)

            return Xi, pi

        # make new graphs with node_0 having X0 & H0
        H_init = self.z_to_h(z)
        X0 = self.h_to_x0(H_init)
        X0, p0 = x0_to_p0(X0)
        G = [dgl.graph(([], [])).to(self.device) for _ in range(len(z))]
        for idx, g in enumerate(G):
            g.add_nodes(1, {'X': X0[idx].unsqueeze(0),
                            'params': p0[idx].unsqueeze})  # add X0 and params-0
        self._propagate(G, 0, H_init)  # compute H0 given X0 & H_init

        # generate nodes/edges 1-by-1 from node_1 upwards
        for vi in range(1, self.n_nodes):

            # generate parameters Xi from Hi-1
            Hg = self._get_hidden(vi - 1)
            Xi = self.h_to_x(Hg)
            Xi, pi = xi_to_pi(Xi)
            for idx, g in enumerate(G):
                g.add_nodes(1, {'X': Xi[idx].unsqueeze(0),
                                'params': pi[idx].unsqueeze})
            Hi = self._propagate(G, vi)
            # generate self-loop-edge
            Ei_self = self.h_to_edge_self(Hi) > 0.5  # (size_batch, 1) boolean
            for idx, g in enumerate(G):
                if Ei_self[idx]:
                    g.add_edges(vi, vi)
            Hi = self._propagate(G, vi)
            # generate in/out-edges
            for vj in range(vi - 1, -1, -1):
                Hj = self._get_hidden(vj)
                # generate in/out edge probabilities
                e_ij = self.h_to_edge(torch.cat([Hi, Hj], -1)) > 0.5  # (size_batch, 2) in/out boolean
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
        sample = torch.randn(n, self.size_Z).to(self.device)
        G = self.decode(sample)
        return G

    def loss(self, q_dist, G_true, w_kld=0.001):
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
            z = q_dist.sample()
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
        l_mse = self.mse(X0[:, :15], X0_true[:, :15]).mean(0).sum()  # see parse_global() in dxdata.py
        l_bce = self.bce(X0[:, 15:17], X0_true[:, 15:17]).mean(0).sum()  # oks,lks - osc & lfo key sync
        l_ce_lfw = self.ce(X0[:, 17:23], p0_true[:, 17]).mean()  # lfw - 6 lfo waveforms
        l_ce_alg = self.ce(X0[:, 23:], p0_true[:, 18]).mean()  # alg - 32 algorithms
        loss_X = l_mse + l_bce + l_ce_lfw + l_ce_alg

        # teacher forcing nodes/edges 1-by-1 and compute/accumulate losses
        loss_E = 0
        for vi in range(1, self.n_nodes):

            # print(vi)
            # print('l_X0:', loss_X)

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
            l_mse = self.mse(Xi[:, :18], Xi_true[:, :18]).mean(0).sum()  # see parse_op() in dxdata.py
            l_bce = self.bce(Xi[:, 18], Xi_true[:, 18]).mean()  # oks,lks - osc & lfo key sync
            l_ce_lc = self.ce(Xi[:, 19:23], pi_true[:, 19]).mean()  # lc - 4 left curves
            l_ce_rc = self.ce(Xi[:, 23:27], pi_true[:, 20]).mean()  # rc - 4 right curves
            loss_X += l_mse + l_bce + l_ce_lc + l_ce_rc

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

            # print()
            # print('l_X', loss_X)
            # print('l_E', loss_E)
            # print()

        kld = kl_divergence(self.p_dist, q_dist).mean(0).sum()

        return loss_X + loss_E + kld * w_kld, loss_X, loss_E, kld * w_kld

    def forward(self, G_true):
        q_dist = self.encode(G_true)
        loss, lp, le, kld = self.loss(q_dist, G_true)
        return loss, lp, le, kld

    def train(self, G_true, epochs, lr=0.0001, checkpoint=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            loss, lp, le, kld = self.forward(G_true)
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}\t l_x: {lp:.4f}\tl_e: {le:.4f}\tkld: {kld:.4f}')
            # if epoch % 5 == 0:
            #     print(epoch)
            #     print(f'l_x: {lp:.4f}\tl_e: {le:.4f}\tkld: {kld:.4f}')
            if checkpoint is not None and epoch % 20 == 0 and epoch != 0:
                torch.save(self.state_dict(), checkpoint)
                print(f'\nCheckpoint [{checkpoint}] saved\n')

        print('Finished Training')
