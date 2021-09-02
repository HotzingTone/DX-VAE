import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
import dgl


class DXVAE(nn.Module):
    def __init__(self, n_nodes=7, n_params=12, size_hidden=512, size_latent=128):
        super().__init__()
        self.n_nodes = n_nodes  # total number of nodes
        self.n_params = n_params  # number of node parameters
        self.size_hidden = size_hidden  # size of hidden state for a node
        self.size_latent = size_latent  # size of latent z
        self.zero_params = torch.zeros(n_params)  # padding params for node_0
        self.zero_hidden = torch.zeros(size_hidden)  # padding hidden state for propagation
        self.hidden = None  # hidden state container
        self.p_dist = Normal(0., 1.)  # prior distribution

        # encoder
        self.propagator_encode = nn.GRUCell(n_params, size_hidden)  # encoder propagator
        self.looper_encode = nn.GRUCell(n_params, size_hidden)  # encoder self-looper
        self.h_to_mu = nn.Linear(size_hidden, size_latent)  # latent mean
        self.h_to_std = nn.Sequential(
            nn.Linear(size_hidden, size_latent),
            nn.Softplus())  # softplus to ensure positive standard deviation

        # decoder
        self.propagator_decode = nn.GRUCell(n_params, size_hidden)  # decoder propagator
        self.looper_decode = nn.GRUCell(n_params, size_hidden)  # decoder self-looper
        self.z_to_h = nn.Sequential(
            nn.Linear(size_latent, size_hidden),
            nn.Tanh()
        )  # latent z to initial hidden state H_init
        self.h_to_params = nn.Sequential(
            nn.Linear(size_hidden, size_hidden * 2),
            nn.ReLU(),
            nn.Linear(size_hidden * 2, n_params),
        )  # node hidden state to parameters
        self.h_to_edge_self = nn.Sequential(
            nn.Linear(size_hidden, size_hidden * 2),
            nn.ReLU(),
            nn.Linear(size_hidden * 2, 1),
            nn.Sigmoid()  # todo: try including params as input
        )
        self.h_to_edge = nn.Sequential(
            nn.Linear(size_hidden * 2, size_hidden * 4),
            nn.ReLU(),
            nn.Linear(size_hidden * 4, 2),
            nn.Sigmoid()
        )  # probabilities for edge vj->vi & vi->vj given (Hvi, Hvj)

        # gated sum for propagation
        self.gate = nn.Sequential(
            nn.Linear(size_hidden * 2, size_hidden),
            nn.Sigmoid()
        )
        self.mapper = nn.Sequential(
            nn.Linear(size_hidden * 2, size_hidden, bias=False),
        )  # disable bias to ensure padded zeros also mapped to zeros

    def _get_hidden(self, v):
        Hv = torch.stack([h[v] for h in self.hidden])
        return Hv

    def _propagate(self, G, v, H_in=None, encode=False):
        if len(G) == 0:
            return
        # pick propagator & looper
        if encode:
            propagator = self.propagator_encode
            looper = self.looper_encode
        else:
            propagator = self.propagator_decode
            looper = self.looper_decode
        # get X and X_loop
        X = torch.stack([g.ndata['params'][v] for g in G])  # (size_batch, n_params)
        X_loop = torch.zeros_like(X)
        for idx, g in enumerate(G):
            if v in g.successors(v):  # if self-loop
                X_loop[idx] = X[idx]  # copy x to x_loop
        # compute H_in
        if H_in is None:
            if encode:  # encode from leaves
                H_forth = torch.stack(
                    [torch.stack(
                        [self.hidden[idx][x]
                         if x in g.predecessors(v)
                         else self.zero_hidden
                         for x in range(v + 1, self.n_nodes)]
                    ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
                H_back = torch.stack(
                    [torch.stack(
                        [self.hidden[idx][x]
                         if x in g.successors(v)
                         else self.zero_hidden
                         for x in range(v + 1, self.n_nodes)]
                    ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
            else:  # decode from root
                H_forth = torch.stack(
                    [torch.stack(
                        [self.hidden[idx][x]
                         if x in g.predecessors(v)
                         else self.zero_hidden
                         for x in range(v - 1, -1, -1)]
                    ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
                H_back = torch.stack(
                    [torch.stack(
                        [self.hidden[idx][x]
                         if x in g.successors(v)
                         else self.zero_hidden
                         for x in range(v - 1, -1, -1)]
                    ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
            H_in = torch.cat([H_forth, H_back], 2)  # (size_batch, n_node - v - 1, size_hidden * 2)
            H_in = (self.gate(H_in) * self.mapper(H_in)).sum(1)  # (size_batch, size_hidden)
        # combine with X
        Hv = propagator(X, H_in)  # (size_batch, size_hidden)
        Hv = looper(X_loop, Hv)  # (size_batch, size_hidden)
        # store Hv into G
        for idx in range(len(G)):
            self.hidden[idx][v] = Hv[idx]

        return Hv

    def encode(self, G):
        self.hidden = [[None] * self.n_nodes for _ in range(len(G))]  # placeholder
        # from the leaf
        H_init = self.zero_hidden.repeat(len(G), 1)
        self._propagate(G, self.n_nodes - 1, H_init, encode=True)
        # down stream
        for v in range(self.n_nodes - 2, -1, -1):
            self._propagate(G, v, encode=True)
        # to the root
        Hg = self._get_hidden(0)  # todo try H_in for root Hg, w/ node_0's params being None
        mu, std = self.h_to_mu(Hg), self.h_to_std(Hg)
        q_dist = Normal(mu, std)
        return q_dist

    def decode(self, z):
        # make new graphs with node_0 having X0 & H0
        G = [dgl.graph(([], [])) for _ in range(len(z))]
        for g in G:
            g.add_nodes(1, {'params': self.zero_params.unsqueeze(0)})  # add X0
        H_init = self.z_to_h(z)
        self._propagate(G, 0, H_init)  # compute H0 given X0 & H_init

        # generate nodes/edges 1-by-1 from 1 upwards
        for vi in range(1, self.n_nodes):
            # generate parameters Xi from Hi-1
            Hg = self._get_hidden(vi - 1)
            Xi = self.h_to_params(Hg)
            torch.sigmoid_(Xi[:, 11])
            for idx, g in enumerate(G):
                g.add_nodes(1, {'params': Xi[idx].unsqueeze(0)})
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
            z = q_dist.sample()
        else:
            z = q_dist.loc
        G = self.decode(z)
        return G

    def generate(self, n):
        self.hidden = [[None] * self.n_nodes for _ in range(n)]  # placeholder
        sample = torch.randn(n, self.size_latent)  # .to(self.get_device()) ?
        G = self.decode(sample)
        return G

    def loss(self, q_dist, G_true, w1=10, w2=0.2, w3=1, w4=1):
        # compute H_init from latent
        if self.training:
            z = q_dist.sample()
        else:
            z = q_dist.loc
        H_init = self.z_to_h(z)
        # make new graphs with node_0 having X0 & H0
        G = [dgl.graph(([], [])) for _ in range(len(z))]
        for g in G:
            g.add_nodes(1, {'params': self.zero_params.unsqueeze(0)})  # add X0
        self._propagate(G, 0, H_init)  # compute H0 given X0 & H_init

        # get true parameters and adjacency matrix
        param_true = torch.stack(
            [g_true.ndata['params']
             for g_true in G_true])  # (size_batch, n_nodes, n_params)
        adj_true = torch.stack(
            [g_true.adj().to_dense()
             for g_true in G_true])  # (size_batch, n_nodes, n_nodes)
        # teacher forcing nodes/edges 1-by-1 and compute/accumulate losses
        loss_params = 0
        loss_edges = 0
        for vi in range(1, self.n_nodes):
            # generate parameters Xi from Hi-1
            Hg = self._get_hidden(vi - 1)
            Xi = self.h_to_params(Hg)
            # teacher forcing true parameters Xi_true to compute Hi
            Xi_true = param_true[:, vi, :]
            for idx, g in enumerate(G):
                g.add_nodes(1, {'params': Xi_true[idx].unsqueeze(0)})
            Hi = self._propagate(G, vi)
            # compute parameter loss
            mse = F.mse_loss(Xi[:, :11], Xi_true[:, :11], reduction='none').mean(0).sum()
            bce = F.binary_cross_entropy(torch.sigmoid(Xi[:, 11]), Xi_true[:, 11], reduction='mean')
            loss_params += mse + bce * w1  # bce for freq mode, having higher weight
            # generate self-loop-edge probability
            Ei_self = self.h_to_edge_self(Hi)
            # teacher forcing Ei_self_true to update Hi
            Ei_self_true = adj_true[:, vi, vi].unsqueeze(1)  # (size_batch, 1)
            for idx, g in enumerate(G):
                if Ei_self_true[idx]:
                    g.add_edges(vi, vi)
            Hi = self._propagate(G, vi)
            # compute self-loop-edge loss
            loss_edges += F.binary_cross_entropy(
                Ei_self, Ei_self_true, reduction='mean')

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
            loss_edges += F.binary_cross_entropy(Ei, Ei_true, reduction='none').mean(0).sum()

        kld = kl_divergence(self.p_dist, q_dist).mean(0).sum()

        return loss_params * w2 + loss_edges * w3 + kld * w4, loss_params * w2, loss_edges * w3, kld * w4

    def forward(self, G_true):
        q_dist = self.encode(G_true)
        loss, lp, le, kld = self.loss(q_dist, G_true)
        return loss, lp, le, kld

    def train(self, G_true, epochs, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, lp, le, kld = self.forward(G_true)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 4:
                # print(epoch, 'loss:', loss)
                print(epoch)
                print('lp:', lp)
                print('le:', le)
                print('kld', kld)

        print('Finished Training')


# class DXVAE_softmax(nn.Module):
#     def __init__(self, n_nodes=7, n_params=13, size_1hot=1049, size_hidden=512, size_latent=128):
#         super().__init__()
#         self.n_nodes = n_nodes  # total number of nodes
#         self.size_1hot = size_1hot  # node parameters size in one-hot
#         self.n_params = n_params  # number of parameters for each node
#         self.size_hidden = size_hidden  # size of hidden state for a node
#         self.size_latent = size_latent  # size of latent z
#         self.zero_params = torch.zeros(n_params).long()  # padding params for node_0
#         self.zero_1hot = torch.zeros(size_1hot)  # padding one_hot params for node_0
#         self.zero_hidden = torch.zeros(size_hidden)  # padding hidden state for propagation
#         self.hidden = None  # hidden state container
#         self.p_dist = Normal(0., 1.)  # prior distribution
#
#         # encode
#         self.propagator_encode = nn.GRUCell(size_1hot, size_hidden)  # encoder propagator
#         self.looper_encode = nn.GRUCell(size_1hot, size_hidden)  # encoder self-looper
#         self.h_to_mu = nn.Linear(size_hidden, size_latent)  # latent mean
#         self.h_to_std = nn.Sequential(
#             nn.Linear(size_hidden, size_latent),
#             nn.Softplus())  # softplus to ensure positive standard deviation
#
#         # decode
#         self.propagator_decode = nn.GRUCell(size_1hot, size_hidden)  # decoder propagator
#         self.looper_decode = nn.GRUCell(size_1hot, size_hidden)  # decoder self-looper
#         self.z_to_h = nn.Sequential(
#             nn.Linear(size_latent, size_hidden),
#             nn.Tanh()
#         )  # latent z to initial hidden state H_init
#         self.h_to_params = nn.Sequential(
#             nn.Linear(size_hidden, size_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(size_hidden * 2, size_1hot),
#         )  # node hidden state to parameters
#         self.h_to_edge_self = nn.Sequential(
#             nn.Linear(size_hidden, size_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(size_hidden * 2, 1),
#             nn.Sigmoid()  # todo: try including params as input
#         )
#         self.h_to_edge = nn.Sequential(
#             nn.Linear(size_hidden * 2, size_hidden * 4),
#             nn.ReLU(),
#             nn.Linear(size_hidden * 4, 2),
#             nn.Sigmoid()
#         )  # probabilities for edge vj->vi & vi->vj given (Hvi, Hvj)
#
#         # gated sum for propagation
#         self.gate = nn.Sequential(
#             nn.Linear(size_hidden * 2, size_hidden),
#             nn.Sigmoid()
#         )
#         self.mapper = nn.Sequential(
#             nn.Linear(size_hidden * 2, size_hidden, bias=False),
#         )  # disable bias to ensure padded zeros also mapped to zeros
#
#     def _get_hidden(self, v):
#         Hv = torch.stack([h[v] for h in self.hidden])
#         return Hv
#
#     def logit_to_prob(self, Xi_logit):
#         Xi_prob = torch.cat([F.log_softmax(Xi_logit[:, :2], 1),
#                              F.log_softmax(Xi_logit[:, 2:34], 1),
#                              F.log_softmax(Xi_logit[:, 34:134], 1),
#                              F.log_softmax(Xi_logit[:, 134:149], 1),
#                              F.log_softmax(Xi_logit[:, 149:249], 1),
#                              F.log_softmax(Xi_logit[:, 249:349], 1),
#                              F.log_softmax(Xi_logit[:, 349:449], 1),
#                              F.log_softmax(Xi_logit[:, 449:549], 1),
#                              F.log_softmax(Xi_logit[:, 549:649], 1),
#                              F.log_softmax(Xi_logit[:, 649:749], 1),
#                              F.log_softmax(Xi_logit[:, 749:849], 1),
#                              F.log_softmax(Xi_logit[:, 849:949], 1),
#                              F.log_softmax(Xi_logit[:, 949:], 1)],
#                             dim=1)
#         return Xi_prob
#
#     def logit_to_params(self, Xi_logit):
#         Xi = torch.stack([Xi_logit[:, :2].argmax(1),  # mode
#                           Xi_logit[:, 2:34].argmax(1),  # coarse
#                           Xi_logit[:, 34:134].argmax(1),  # fine
#                           Xi_logit[:, 134:149].argmax(1),  # tune
#                           Xi_logit[:, 149:249].argmax(1),  # R1
#                           Xi_logit[:, 249:349].argmax(1),  # R2
#                           Xi_logit[:, 349:449].argmax(1),  # R3
#                           Xi_logit[:, 449:549].argmax(1),  # R4
#                           Xi_logit[:, 549:649].argmax(1),  # L1
#                           Xi_logit[:, 649:749].argmax(1),  # L2
#                           Xi_logit[:, 749:849].argmax(1),  # L3
#                           Xi_logit[:, 849:949].argmax(1),  # L4
#                           Xi_logit[:, 949:].argmax(1)]).T  # gain
#         return Xi
#
#     def params_to_1hot(self, Xi):
#         Xi_1hot = torch.cat([F.one_hot(Xi[:, 0], 2),  # mode
#                              F.one_hot(Xi[:, 1], 32),  # coarse
#                              F.one_hot(Xi[:, 2], 100),  # fine
#                              F.one_hot(Xi[:, 3], 15),  # tune
#                              F.one_hot(Xi[:, 4], 100),  # R1
#                              F.one_hot(Xi[:, 5], 100),  # R2
#                              F.one_hot(Xi[:, 6], 100),  # R3
#                              F.one_hot(Xi[:, 7], 100),  # R4
#                              F.one_hot(Xi[:, 8], 100),  # L1
#                              F.one_hot(Xi[:, 9], 100),  # L2
#                              F.one_hot(Xi[:, 10], 100),  # L3
#                              F.one_hot(Xi[:, 11], 100),  # L4
#                              F.one_hot(Xi[:, 12], 100)],  # gain
#                             dim=1)
#         return Xi_1hot.float()
#
#     def _propagate(self, G, v, H_in=None, encode=False):
#         if len(G) == 0:
#             return
#         # pick propagator & looper
#         if encode:
#             propagator = self.propagator_encode
#             looper = self.looper_encode
#         else:
#             propagator = self.propagator_decode
#             looper = self.looper_decode
#         # get X_1hot and X_1hot_loop
#         X_1hot = torch.stack([g.ndata['params_1hot'][v] for g in G])  # (size_batch, size_1hot)
#         X_1hot_loop = torch.zeros_like(X_1hot)
#         for idx, g in enumerate(G):
#             if v in g.successors(v):  # if self-loop
#                 X_1hot_loop[idx] = X_1hot[idx]  # copy x to x_loop
#         # compute H_in
#         if H_in is None:
#             if encode:  # encode from leaves
#                 H_forth = torch.stack(
#                     [torch.stack(
#                         [self.hidden[idx][x]
#                          if x in g.predecessors(v)
#                          else self.zero_hidden
#                          for x in range(v + 1, self.n_nodes)]
#                     ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
#                 H_back = torch.stack(
#                     [torch.stack(
#                         [self.hidden[idx][x]
#                          if x in g.successors(v)
#                          else self.zero_hidden
#                          for x in range(v + 1, self.n_nodes)]
#                     ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
#             else:  # decode from root
#                 H_forth = torch.stack(
#                     [torch.stack(
#                         [self.hidden[idx][x]
#                          if x in g.predecessors(v)
#                          else self.zero_hidden
#                          for x in range(v - 1, -1, -1)]
#                     ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
#                 H_back = torch.stack(
#                     [torch.stack(
#                         [self.hidden[idx][x]
#                          if x in g.successors(v)
#                          else self.zero_hidden
#                          for x in range(v - 1, -1, -1)]
#                     ) for idx, g in enumerate(G)])  # (size_batch, n_node - v - 1, size_hidden)
#             H_in = torch.cat([H_forth, H_back], 2)  # (size_batch, n_node - v - 1, size_hidden * 2)
#             H_in = (self.gate(H_in) * self.mapper(H_in)).sum(1)  # (size_batch, size_hidden)
#         # combine with X_1hot
#         Hv = propagator(X_1hot, H_in)  # (size_batch, size_hidden)
#         Hv = looper(X_1hot_loop, Hv)  # (size_batch, size_hidden)
#         # store Hv into G
#         for idx in range(len(G)):
#             self.hidden[idx][v] = Hv[idx]
#
#         return Hv
#
#     def encode(self, G):
#         self.hidden = [[None] * self.n_nodes for _ in range(len(G))]  # placeholder
#         # from the leaf
#         H_init = self.zero_hidden.repeat(len(G), 1)
#         self._propagate(G, self.n_nodes - 1, H_init, encode=True)
#         # down stream
#         for v in range(self.n_nodes - 2, -1, -1):
#             self._propagate(G, v, encode=True)
#         # to the root
#         Hg = self._get_hidden(0)  # todo try H_in for root Hg, w/ node_0's params being None
#         mu, std = self.h_to_mu(Hg), self.h_to_std(Hg)
#         q_dist = Normal(mu, std)
#         return q_dist
#
#     def decode(self, z):
#         # make new graphs with node_0 having X0, X0_1hot & H0
#         G = [dgl.graph(([], [])) for _ in range(len(z))]
#         for g in G:
#             g.add_nodes(1, {'params': self.zero_params.unsqueeze(0),
#                             'params_1hot': self.zero_1hot.unsqueeze(0)})  # add X0, X0_1hot
#         H_init = self.z_to_h(z)
#         self._propagate(G, 0, H_init)  # compute H0 given X0_1hot & H_init
#
#         # generate nodes/edges 1-by-1 from 1 upwards
#         for vi in range(1, self.n_nodes):
#             # generate parameters Xi from Hi-1
#             Hg = self._get_hidden(vi - 1)
#             Xi_logit = self.h_to_params(Hg)
#             Xi = self.logit_to_params(Xi_logit)
#             Xi_1hot = self.params_to_1hot(Xi)
#             for idx, g in enumerate(G):
#                 g.add_nodes(1, {'params': Xi[idx].unsqueeze(0),
#                                 'params_1hot': Xi_1hot[idx].unsqueeze(0)})
#             Hi = self._propagate(G, vi)
#
#             # generate self-loop-edge
#             Ei_self = self.h_to_edge_self(Hi) > 0.5  # (size_batch, 1) boolean
#             for idx, g in enumerate(G):
#                 if Ei_self[idx]:
#                     g.add_edges(vi, vi)
#             Hi = self._propagate(G, vi)
#             # generate in/out-edges
#             for vj in range(vi - 1, -1, -1):
#                 Hj = self._get_hidden(vj)
#                 # generate in/out edge probabilities
#                 e_ij = self.h_to_edge(torch.cat([Hi, Hj], -1)) > 0.5  # (size_batch, 2) in/out boolean
#                 for idx, g in enumerate(G):
#                     if e_ij[idx, 0]:  # in-edge, j->i
#                         g.add_edges(vj, vi)
#                     if e_ij[idx, 1]:  # out-edge, i->j
#                         g.add_edges(vi, vj)
#                 Hi = self._propagate(G, vi)
#
#         return G
#
#     def encode_decode(self, G, stochastic=False):
#         q_dist = self.encode(G)
#         if stochastic:
#             z = q_dist.sample()
#         else:
#             z = q_dist.loc
#         G = self.decode(z)
#         return G
#
#     def generate(self, n):
#         self.hidden = [[None] * self.n_nodes for _ in range(n)]  # placeholder
#         sample = torch.randn(n, self.size_latent)  # .to(self.get_device()) ?
#         G = self.decode(sample)
#         return G
#
#     def loss(self, q_dist, G_true):
#         # compute H_init from latent
#         if self.training:
#             z = q_dist.sample()
#         else:
#             z = q_dist.loc
#         H_init = self.z_to_h(z)
#         # make new graphs with node_0 having X0, X0_1hot & H0
#         size_batch = len(z)
#         G = [dgl.graph(([], [])) for _ in range(size_batch)]
#         for g in G:
#             g.add_nodes(1, {  # 'params': self.zero_params.unsqueeze(0),
#                 'params_1hot': self.zero_1hot.unsqueeze(0)})  # add X0, X0_1hot
#         self._propagate(G, 0, H_init)  # compute H0 given X0_1hot & H_init
#
#         # get true parameters and adjacency matrix
#         params_true = torch.stack(
#             [g_true.ndata['params']
#              for g_true in G_true])  # (size_batch, n_nodes, n_params)
#         params_1hot_true = torch.stack(
#             [g_true.ndata['params_1hot']
#              for g_true in G_true])  # (size_batch, n_nodes, size_1hot)
#         adj_true = torch.stack(
#             [g_true.adj().to_dense()
#              for g_true in G_true])  # (size_batch, n_nodes, n_nodes)
#
#         # teacher forcing nodes/edges 1-by-1 and compute/accumulate losses
#         loss_params = 0
#         loss_edges = 0
#         for vi in range(1, self.n_nodes):
#             # generate param probabilities Xi_prob from Hi-1
#             Hg = self._get_hidden(vi - 1)
#             Xi_logit = self.h_to_params(Hg)
#             Xi_prob = self.logit_to_prob(Xi_logit)
#             # teacher forcing true one-hot params Xi_1hot_true to compute Hi
#             Xi_1hot_true = params_1hot_true[:, vi, :]  # (graph, node, 1hot)
#             for idx, g in enumerate(G):
#                 g.add_nodes(1, {'params_1hot': Xi_1hot_true[idx].unsqueeze(0)})
#             Hi = self._propagate(G, vi)
#             # compute parameter loss
#             loss_params += -(Xi_prob * Xi_1hot_true).sum(1).mean(0)  # negative log-likelihood
#             # generate self-loop-edge probability
#             Ei_self = self.h_to_edge_self(Hi)
#             # teacher forcing Ei_self_true to update Hi
#             Ei_self_true = adj_true[:, vi, vi].unsqueeze(1)  # (size_batch, 1)
#             for idx, g in enumerate(G):
#                 if Ei_self_true[idx]:
#                     g.add_edges(vi, vi)
#             Hi = self._propagate(G, vi)
#             # compute self-loop-edge loss
#             loss_edges += F.binary_cross_entropy(
#                 Ei_self, Ei_self_true, reduction='mean')
#
#             # consider in/out-edge probabilities
#             Ei = []
#             adj_in_true = adj_true[:, :vi, vi].unsqueeze(2)  # (size_batch, vi, 1)
#             adj_out_true = adj_true[:, vi, :vi].unsqueeze(2)  # (size_batch, vi, 1)
#             Ei_true = torch.cat([adj_in_true, adj_out_true], 2)  # (size_batch, vi, 2) in/out adj
#             # count edges from edge_i_i-1 ... edge_i_0
#             for vj in range(vi - 1, -1, -1):
#                 Hj = self._get_hidden(vj)
#                 # generate in/out edge probabilities
#                 e_ij = self.h_to_edge(torch.cat([Hi, Hj], -1)).unsqueeze(1)  # (size_batch, 1, 2) in/out prob
#                 Ei.append(e_ij)
#                 # teacher forcing true edges and update Hi
#                 for idx, g in enumerate(G):
#                     if adj_in_true[idx, vj]:
#                         g.add_edges(vj, vi)
#                     if adj_out_true[idx, vj]:
#                         g.add_edges(vi, vj)
#                 Hi = self._propagate(G, vi)
#             # resorted list as edge_i_0 ... edge_i_i-1
#             Ei.reverse()
#             Ei = torch.cat(Ei, 1)  # (size_batch, vi, 2) in/out prob
#             # compute out/in-edge losses
#             loss_edges += F.binary_cross_entropy(Ei, Ei_true, reduction='none').mean(0).sum()
#
#         kld = kl_divergence(self.p_dist, q_dist).mean(0).sum()
#
#         return loss_params + loss_edges + kld, loss_params, loss_edges, kld
#
#     def forward(self, G_true):
#         q_dist = self.encode(G_true)
#         loss, lp, le, kld = self.loss(q_dist, G_true)
#         return loss, lp, le, kld
#
#     def train(self, G_true, epochs, lr=0.001):
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             loss, lp, le, kld = self.forward(G_true)
#             loss.backward()
#             optimizer.step()
#             if epoch % 5 == 4:
#                 # print(epoch, 'loss:', loss)
#                 print(epoch)
#                 print('lp:', lp)
#                 print('le:', le)
#                 print('kld', kld)
#
#         print('Finished Training')
