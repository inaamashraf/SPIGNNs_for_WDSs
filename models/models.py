import torch
from torch_scatter import scatter
from utils.utils import *
from models.layers import *
from torch.nn import ModuleList, L1Loss, Module
from torch_geometric.nn.dense.linear import Linear
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PI_GNN(Module):
    """
        Our model using local GNN layers learning from a global Physics-Informed Algorithm.
    """
    def __init__(self, M_n, out_dim, M_e, M_l=128, aggr='max', \
                    I=5, bias=False, num_layers=2, n_iter=10):
        super(PI_GNN, self).__init__()

        """ Number of GNN layers. """
        self.I = I   
        """ Minimum number of iterations. """           
        self.n_iter = n_iter    

        """ 
            Linear layers alpha and beta used to embed node 
            and edge features respectively.
        """
        self.node_in = Linear(M_n, M_l, bias=bias)
        self.edge = Linear(M_e, M_l, bias=bias)   

        """ The MLP lambda used to estimate the flows q_hat. """
        self.flows_latent = MLP([3*M_l, M_l, M_l, out_dim], bias=bias, activ=None)   

        """ GNN layers """
        self.gcn_aggrs = ModuleList()
        for _ in range(I):
            gcn = GNN_Layer(M_l, M_l, M_l, aggr=aggr, bias=bias, num_layers=num_layers)
            self.gcn_aggrs.append(gcn)        

    def forward(self, data, r_iter=5, zeta=1e-32, epoch=300, demand=None, _test=False):

        """ Reading data. """
        data = data.to(device)
        x, edge_index, r = data.x, data.edge_index, data.edge_attr[:, 0:1]   
        self.n_nodes = int(data.num_nodes / data.num_graphs)
        self.n_edges = int(data.num_edges / data.num_graphs)        
        
        """ True demands d_star. """
        self.d_star = x[:, 1:2]             

        """ 
            True heads h_star where only reservoirs have values 
            and all other values are zeros. 
        """
        self.h_star = x[:, 0:1]                  
        
        """ Creating the reservoir mask to be used in the loss function. """
        self.reservoir_mask = self.h_star != 0

        """ Initializing h as self.h_star. """
        h = self.h_star.clone()

        """ Computing initial flows (q_hat_(0)) and demands (d_hat_(0)). """
        self.d_hat, self.q_hat = \
                compute_net_flows(
                            h, 
                            r, 
                            edge_index, 
                            zeta = zeta
                            )    
        
        """ Initializing q_tilde_(0) the same as q_hat_(0). """
        self.q_tilde = self.q_hat.clone()

        """
            Specifying the additional number of iterations.
            These are chosen randomly during training but are 
            set to maximum specified value in evaluation.
        """
        if self.training:
            K = self.n_iter + np.random.randint(0, r_iter)
        else:
            K = self.n_iter + r_iter

        """ Performing K iteration. """
        for _ in range(K):

            """____ f_1 ((D,Q), Theta)___"""

            """ 
                Embedding node and edge features using linear layers 
                alpha and beta respectively.
            """
            g = self.node_in(torch.selu(torch.cat((self.d_hat, self.d_star), dim=-1)))
            z = self.edge(torch.selu(torch.cat((self.q_tilde, self.q_hat), dim=-1)))

            """ 
                Multiple (I) GNN layers 
            """
            for gcn in self.gcn_aggrs:
                g, z = gcn(g, edge_index, z)

            """ Estimating flows q_hat using the MLP lambda. """
            sndr_g = g[edge_index[0, :], :]
            rcvr_g = g[edge_index[1, :], :]
            self.q_hat = self.q_hat + self.flows_latent(torch.selu(torch.cat((sndr_g, rcvr_g, z), dim=-1)))

            """ Adjusting the flows q_hat for directionality. """
            self.q_hat = torch.stack(self.q_hat.split(self.n_edges))
            self.q_hat_in = self.q_hat[:, : self.n_edges//2, :] 
            self.q_hat = torch.cat((self.q_hat_in, self.q_hat_in * -1), dim=1)
            self.q_hat = torch.cat((*self.q_hat,), dim=0)   

            """ Computing estimated demands d_hat using hydraulic principle (eq. 3). """
            self.d_hat = scatter(self.q_hat, dim=0, index=edge_index[1:2, :].T, reduce='add')
            

            """____ f_2 (h, q_hat)___"""

            """ Reconstructing heads using our physics-informed algorithm (eq. 6 & 7). """            
            J = self.I * self.n_iter
            self.h_tilde = construct_heads(
                            J, 
                            self.h_star.clone(), 
                            self.q_hat, 
                            r,
                            edge_index,
                            zeta = zeta
                            ) 

            """ Computing flows q_tilde and demand d_tilde using hydraulic principle (eq. 8). """
            self.d_tilde, self.q_tilde = \
                compute_net_flows(
                            self.h_tilde, 
                            r, 
                            edge_index, 
                            zeta = zeta
                            )    
           
        return self.h_tilde        

    def loss(self, rho=0.1, delta=0.1):
        """ L1 Loss """       
        l1loss = L1Loss(reduction='mean')

        """ Computing loss between true demands d_star and estimated demands by f_1 i.e. d_hat. """     
        self.loss_d_hat = l1loss(self.d_hat[~self.reservoir_mask], self.d_star[~self.reservoir_mask]) 

        """ Computing loss between true demands d_star and estimated demands at the end i.e. d_tilde. """     
        self.loss_d_tilde = l1loss(self.d_tilde[~self.reservoir_mask], self.d_star[~self.reservoir_mask]) 

        """ Computing loss between estimated flows q_hat (f_1) and q_tilde (f). """     
        self.loss_q = l1loss(self.q_hat, self.q_tilde) 

        """ Summing the losses. """
        _loss =  self.loss_d_hat + rho * self.loss_d_tilde + delta * self.loss_q

        return _loss
    



class SPI_GNN(Module):
    """
        Our model using local GNN layers learning from a global Physics-Informed Algorithm.
    """
    def __init__(self, M_n, out_dim, M_e, M_l=128, aggr='max', dia=1,\
                    I=5, bias=False, num_layers=2, n_iter=10):
        super(SPI_GNN, self).__init__()

        """ Number of GNN layers. """
        self.I = I   
        """ Minimum number of iterations. """           
        self.n_iter = n_iter    
        """ Latent layer dimension. """           
        self.M_l = M_l
        """ Diameter of the WDS. """                   
        self.dia = dia

        """ 
            Since we only use MLPs with single layer, 
            the MLPs alpha, beta, lambda and phi are defined as linear layers.
        """
        self.node_in = Linear(M_n, M_l, bias=bias)
        self.edge = Linear(M_e, M_l, bias=bias)   
        self.z_latent = Linear(3*M_l, M_l, bias=bias)   
        self.flows_latent = Linear(2*M_l, out_dim, bias=bias)   

        """ GNN layers """
        self.gcn_aggrs = ModuleList()
        for _ in range(I):
            gcn = SGNN_Layer(M_l, M_l, M_l, aggr=aggr, bias=bias, num_layers=num_layers)
            self.gcn_aggrs.append(gcn)     

    def forward(self, data, r_iter=5, zeta=1e-24, epoch=1500, demand=None):

        """ Reading data. """
        data = data.to(device)
        x, edge_index, r = data.x, data.edge_index, data.edge_attr[:, 0:1]  

        self.prv_mask_nodes = x[:, 2:3]
        self.pump_mask_nodes = x[:, 3:4]
        self.prv_mask_edges = data.edge_attr[:, 2:3] 
        self.pump_mask_edges = data.edge_attr[:, 3:4] 
        self.pump_curve_coefs = data.edge_attr[:, 4:8]

        self.edge_direct_mask = data.edge_attr[:, 1:2] 
        self.batch_size = data.num_graphs
        self.n_nodes = int(data.num_nodes / self.batch_size)
        self.n_edges = int(data.num_edges / self.batch_size)        
        self.epoch = epoch

        self.pump_mask_edges_dir = self.pump_mask_edges[self.edge_direct_mask[:,0] == 1, :]

        """ True demands d_star. """
        self.d_star = x[:, 1:2]

        """ 
            True heads h_star where only reservoirs have values 
            and all other values are zeros. 
        """
        self.h_star = x[:, 0:1].clone()                  
        
        """ Creating the reservoir mask to be used in the loss function. """
        self.reservoir_mask = x[:, 4:5].bool()

        """ Initializing demands (d_hat_(0)), flows (q_hat_(0)) and head_losses (l_hat_(0)). """
        self.d_hat, self.q_hat, self.l_hat = torch.zeros_like(self.d_star), torch.zeros_like(r), torch.zeros_like(r)  
        
        """ Initializing demands (d_tilde_(0)), flows (q_tilde_(0)) and head_losses (l_tilde_(0)). """
        self.d_tilde, self.q_tilde, self.l_tilde = torch.zeros_like(self.d_star), torch.zeros_like(r), torch.zeros_like(r)  

        """ Initializing intermediate flows. """
        q_hat_dir = torch.zeros_like(r[self.edge_direct_mask == 0][:, None]) 

        """
            Specifying the additional number of iterations.
            These are chosen randomly during training but are 
            set to maximum specified value in evaluation.
        """
        if self.training:
            K = self.n_iter + np.random.randint(0, r_iter)
        else:
            K = self.n_iter + r_iter

        """ Performing K iteration. """
        for k in range(K):

            """____ f_1 ((D,Q), Theta)___"""

            """ 
                Embedding node and edge features using linear layers 
                alpha and beta respectively.
            """
            g = (self.node_in(torch.cat((self.d_hat, self.d_star, self.reservoir_mask.float()), dim=-1)))
            z = self.edge(torch.cat((self.q_tilde, self.q_hat), dim=-1))

            """ 
                Multiple (I) GNN layers 
            """
            for gcn in self.gcn_aggrs:
                g, z = gcn(g, edge_index, z, edge_mask=self.edge_direct_mask)

            """ Estimating directed flows q_hat_dir. """
            sndr_g = g[edge_index[0, :], :]
            rcvr_g = g[edge_index[1, :], :]
            z_bar = self.z_latent(torch.cat((sndr_g, rcvr_g, z), dim=-1))
            q_hat_dir = q_hat_dir + self.flows_latent(torch.cat((z_bar[self.edge_direct_mask[:,0] == 0, :], z_bar[self.edge_direct_mask[:,0] == 1, :]), dim=-1))

            """ Ensuring uni-directional flows through pumps. """
            q_hat_dir[self.pump_mask_edges_dir[:,0] == 1, :] = torch.minimum(q_hat_dir[self.pump_mask_edges_dir[:,0] == 1, :], torch.zeros_like(q_hat_dir[self.pump_mask_edges_dir[:,0] == 1, :]))
            q_hat_dir[self.pump_mask_edges_dir[:,0] == 2, :] = 0

            """ Setting the other half of flows equal to the negative of the computed q_hat_dir. """
            q_hat_dir_bi = torch.zeros_like(self.q_hat)
            q_hat_dir_bi[self.edge_direct_mask == 0] = q_hat_dir[:,0]
            q_hat_dir_bi[self.edge_direct_mask == 1] = q_hat_dir[:,0] * -1
            self.q_hat = q_hat_dir_bi * (1 - self.edge_direct_mask) + q_hat_dir_bi * self.edge_direct_mask

            """ Computing estimated demands d_hat using the law of conservation of mass. """
            self.d_hat = scatter(self.q_hat, dim=0, index=edge_index[1:2, :].T, reduce='add')

            """____ f_2 (h, q_hat)___"""

            """ Reconstructing heads using our physics-informed algorithm. 
                f_2 is only run for T-K iterations.
            """            
            if ((k+1) * self.I) >= self.dia:

                self.h_tilde, self.l_hat, J \
                    = construct_heads_pp(
                                        h = self.h_star.clone(), 
                                        q = self.q_hat.clone(), 
                                        r = r,
                                        edge_index = edge_index,
                                        m_n_prv = self.prv_mask_nodes,
                                        m_e_pump = self.pump_mask_edges,
                                        pump_ccs = self.pump_curve_coefs,
                                        zeta = zeta,
                                        reservoir_mask = self.reservoir_mask,
                                        ) 

                """ Computing flows q_tilde and demand d_tilde using hydraulic principle (eq. 8). """
                self.d_tilde, self.q_tilde, self.l_tilde \
                    = compute_net_flows_pp(
                                            h = self.h_tilde, 
                                            r = r, 
                                            edge_index = edge_index,
                                            d = self.d_star,
                                            m_e_prv = self.prv_mask_edges, 
                                            m_e_pump = self.pump_mask_edges,
                                            pump_ccs = self.pump_curve_coefs,
                                            zeta = zeta
                                            )  

                """ Once again, ensuring uni-directional flows through pumps. """
                self.q_hat[self.pump_mask_edges[:,0] == 1, :] = self.q_tilde[self.pump_mask_edges[:,0] == 1, :]
                self.q_hat[self.pump_mask_edges[:,0] == -1, :] = self.q_tilde[self.pump_mask_edges[:,0] == -1, :]
                self.d_hat = scatter(self.q_hat, dim=0, index=edge_index[1:2, :].T, reduce='add')

        return self.h_tilde        

    def loss(self, rho=0.1, delta=0.1):
        """ L1 Loss """       
        l1loss = L1Loss(reduction='mean')

        """ Computing loss between true demands d_star and estimated demands by f_1 i.e. d_hat. """     
        self.loss_d_hat = l1loss(self.d_hat[~self.reservoir_mask], self.d_star[~self.reservoir_mask])

        """ Computing loss between true demands d_star and estimated demands by f_2 i.e. d_tilde. """     
        self.loss_d_tilde = l1loss(self.d_tilde[~self.reservoir_mask], self.d_star[~self.reservoir_mask])

        """ Computing loss between estimated flows q_hat (f_1) and q_tilde (f_2). """     
        self.loss_q = l1loss(self.q_hat, self.q_tilde) 

        """ Summing the losses. """
        _loss =  self.loss_d_hat + rho * self.loss_d_tilde + delta * self.loss_q

        return _loss

