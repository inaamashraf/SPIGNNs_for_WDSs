import torch
import datetime, copy, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import wntr
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
from python_calamine import load_workbook
from torch_scatter import scatter

class WaterFutureScenario:
    """
    Object that loads the Waterfutures Scenario.
    ...    

    Methods
    -------
    
    get_demands():
        Return demands of the scenario.    
    get_pressures():
        Return pressures of the scenario.
    
    """

    def __init__(self, directory):
        
        """
        directory : str - directory of the measurements.            
        """
        
        self.__directory = directory
        self.__time_start = None
        self.__time_step_seconds = None
        self.__leaks_path = directory
        self.__demands = None
        self.__wntr_demands = None
        self.__flows = None
        self.__velocities = None
        self.__pressures = None
        self.__leaks = None
        self.__leaks_details = None

    def __load_time_info(self):
        tmp_df = self.get_pressures()
        self.__time_start = tmp_df.index[0]
        time_step = tmp_df.index[1]
        self.__time_step_seconds = (time_step - self.__time_start).total_seconds()

    def __timestamp_to_idx(self, timestamp):
        if self.__time_start is None or self.__time_step_seconds is None:
            self.__load_time_info()
        
        date_format = '%Y-%m-%d %H:%M:%S'
        cur_time = datetime.datetime.strptime(timestamp, date_format)
        return int((cur_time - self.__time_start).total_seconds() / self.__time_step_seconds)

    def __load_velocities(self):
        ''' Load the velocities for this scenario and save in internal variable. '''
        path_to_velocities = self.__directory

        # tmp_df = pd.read_csv(path_to_velocities)
        try:
            tmp_df = pd.read_csv(path_to_velocities)
        except:
            recs: list[list] = load_workbook(path_to_velocities).get_sheet_by_index(6).to_python()
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
        
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__velocities = tmp_df.astype("float32") #* 1e3
       
    def get_velocities(self):
        if self.__velocities is None:
            self.__load_velocities()
        return self.__velocities.copy() 

    def __load_flows(self):
        ''' Load the flows for this scenario and save in internal variable. '''
        path_to_flows = self.__directory

        try:
            tmp_df = pd.read_csv(path_to_flows)
        except:
            recs: list[list] = load_workbook(path_to_flows).get_sheet_by_index(2).to_python()
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
       
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__flows = tmp_df.astype("float32") #* 1e3
       
    def get_flows(self):
        if self.__flows is None:
            self.__load_flows()
        return self.__flows.copy() 

    def __load_wntr_demands(self):
        ''' Load the demands for this scenario and save in internal variable. '''
        path_to_wntr_demands = self.__directory

        try:
            tmp_df = pd.read_csv(path_to_wntr_demands)
        except:
            recs: list[list] = load_workbook(path_to_wntr_demands).get_sheet_by_index(1).to_python()
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
        
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__wntr_demands = tmp_df.astype("float32") #* 1e3
       
    def get_wntr_demands(self):
        if self.__wntr_demands is None:
            self.__load_wntr_demands()
        return self.__wntr_demands.copy()   

    def __load_demands(self):
        ''' Load the demands for this scenario and save in internal variable. '''
        path_to_demands = self.__directory

        try:
            tmp_df = pd.read_csv(path_to_demands)
        except:
            recs: list[list] = load_workbook(path_to_demands).get_sheet_by_index(5).to_python()
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])
        
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        self.__demands = tmp_df.astype("float32") #* 1e3
       
    def get_demands(self):
        if self.__demands is None:
            self.__load_demands()
        return self.__demands.copy()   

    def __load_pressures(self):
        path_to_pressures = self.__directory
        try:
            tmp_df = pd.read_csv(path_to_pressures)
        except:
            recs: list[list] = load_workbook(path_to_pressures).get_sheet_by_index(4).to_python()
            tmp_df = pd.DataFrame.from_records(recs, coerce_float=True)
            tmp_df.columns = tmp_df.iloc[0]
            tmp_df = tmp_df.drop(tmp_df.index[0])

        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])#, unit='s')
        tmp_df = tmp_df.set_index("Timestamp")
        self.__pressures = tmp_df.astype("float32")
    
    def get_pressures(self):
        if self.__pressures is None:
            self.__load_pressures()
        return self.__pressures.copy()
    
    def __load_leaks_details(self):
        if self.__leaks_path is None:
            self.__leaks_details =  pd.DataFrame.empty
            return

        file_leaks = os.path.join(self.__leaks_path, "Leak_0.xlsx") 
        try:
            self.__leaks_details = pd.read_csv(file_leaks, index_col='Description')
        except:
            self.__leaks_details = pd.read_excel(file_leaks, index_col='Description')
        # Convert timestamps to indices
        self.__leaks_details.loc['Leak Start'] = self.__leaks_details.loc['Leak Start'].apply(self.__timestamp_to_idx)
        self.__leaks_details.loc['Leak End'] = self.__leaks_details.loc['Leak End'].apply(self.__timestamp_to_idx)
        self.__leaks_details.loc['Peak Time'] = self.__leaks_details.loc['Peak Time'].apply(self.__timestamp_to_idx)

    def get_leaks_details(self):
        if self.__leaks_details is None:
            self.__load_leaks_details()
        return self.__leaks_details.copy()


def normalize(X, dim=None, a=0, b=1, _min=None, _max=None):
    """ 
        Method to perform Min-Max normalization of the data.
    """
    E = 1e-32
    
    if dim is not None:
        if _min is None:
            _min = X.min(dim=dim)[0].unsqueeze(dim)
        if _max is None:    
            _max = X.max(dim=dim)[0].unsqueeze(dim)
        X = (b - a) * ((X - _min) / (_max - _min + E)) + a
    else:
        if _min is None:
            _min = X.min()
        if _max is None:    
            _max = X.max()
        X = (b - a) * ((X - _min) / (_max - _min + E)) + a

    return X


def convert_to_bi_edges(edge_index, edge_attr=None, mask=None):    
    """ 
        Method to convert directed edges to bi-directional edges.
    """    
    if mask is not None:
        edge_index_swap = edge_index[:, ~mask].clone()
        edge_index = edge_index[:, ~mask]
    else:
        edge_index_swap = edge_index.clone()
    edge_index_swap_copy = edge_index_swap.clone()
    edge_index_swap[0,:] = edge_index_swap_copy[1,:]
    edge_index_swap[1,:] = edge_index_swap_copy[0,:]   
    edge_index_bi = torch.cat([edge_index, edge_index_swap], dim=-1)
    if edge_attr is not None:
        if mask is not None:
            # edge_attr_bi = torch.cat([edge_attr, edge_attr[:, ~mask, :]], dim=1)
            edge_attr_bi = torch.cat([edge_attr[:, ~mask, :], edge_attr[:, ~mask, :]], dim=1)
        else:
            edge_attr_bi = torch.cat([edge_attr, edge_attr], dim=1)
        return edge_index_bi, edge_attr_bi
    else:
        return edge_index_bi
    

def relabel_nodes(node_indices, edge_indices):
    """ Relabels node and edge indices starting from 0. """

    node_idx = np.where(node_indices >= 0)[0]
    edge_idx = copy.deepcopy(edge_indices)
    for idx, index in zip(node_idx, node_indices):
        edge_idx = np.where(edge_indices == index, idx, edge_idx)
    edge_idx = torch.tensor(edge_idx, dtype=int)

    return node_idx, edge_idx



def compute_net_flows(h, r, edge_index, zeta=1e-32, dia=None, velocity=False):
    """ 
        Computing flows and demands from heads using 
        hydraulic principles. 
    """
        
    sndr_node_attr = h[edge_index[0, :], :] 
    rcvr_node_attr = h[edge_index[1, :], :] 
    h_l = sndr_node_attr - rcvr_node_attr 
    h_lr = h_l / r
    h_lr = torch.nan_to_num(h_lr, nan=0, posinf=0, neginf=0)
    q = torch.pow(h_lr.abs() + zeta, 1/1.852) * torch.sign(h_lr)

    d = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce='add', out=torch.zeros_like(h)) 

    return d, q



def compute_net_flows_pp(h, r, edge_index, d, m_e_prv, m_e_pump, pump_ccs, zeta=1e-32):
    """ 
        Computing flows and demands from heads using 
        hydraulic principles for pipes, pumps and PRVs. 
    """
        
    """ For pipes. """
    sndr_node_attr = h[edge_index[0, :], :] 
    rcvr_node_attr = h[edge_index[1, :], :] 
    h_l = sndr_node_attr - rcvr_node_attr 
    h_lr = h_l / r

    h_lr[h_lr == 0] = zeta
    q = torch.pow(h_lr.abs(), 1/1.852) * torch.sign(h_lr)

    """ For pumps. """
    if (m_e_pump == 1).sum() != 0:
        pump_ccs_orig = pump_ccs.clone()
        pump_ccs = pump_ccs_orig[m_e_pump[:,0] == 1, :]
        q_pumps_x = ((h_l[m_e_pump[:,0] == 1, :] + pump_ccs[:, 3:4]**2 * pump_ccs[:, 0:1]) * pump_ccs[:, 3:4]**pump_ccs[:, 2:3]) / \
                    (pump_ccs[:, 3:4]**2 * pump_ccs[:, 1:2])
        q_pumps = torch.pow(q_pumps_x.relu() + zeta, 1/pump_ccs[:, 2:3])

        q[m_e_pump[:,0] == 1, :] = q_pumps
        q[m_e_pump[:,0] == -1, :] = q_pumps * -1

        q[m_e_pump[:,0] == 2, :] = 0.
        q[m_e_pump[:,0] == -2, :] = 0.

    """ For PRVs. """
    if (m_e_prv > 0).sum() != 0:
        q[m_e_prv != 0] = 0.
        q_sum = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce='add', out=torch.zeros_like(h)) 
        q_sum_in = -1 * (q_sum[edge_index[1, :], :] - d[edge_index[1, :], :]) 
        q[m_e_prv != 0] = q_sum_in[m_e_prv != 0]

    """ Computing demands from flows. """
    d = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce='add', out=torch.zeros_like(h)) 

    return d, q, h_lr



def construct_heads(J, h, q, r, edge_index, zeta=1e-32):   
    """ 
        Reconstructing heads using our physics-informed algorithm. 
    """            
            
    q_x = (torch.pow(q.abs() + zeta, 1.852) * torch.sign(q)) + zeta
    h_l = q_x * r
    for _ in range(J):
        sndr_node_attr = h[edge_index[0, :], :]
        h = scatter(sndr_node_attr - torch.relu(h_l), dim=0, index=edge_index[1:2, :].T, reduce='max', out=h.clone())

    return h



def construct_heads_pp(h, q, r, edge_index, 
                        m_n_prv, m_e_pump, pump_ccs, 
                        reservoir_mask, zeta=1e-24, dia=20):   
    """ 
        Reconstructing heads using our physics-informed algorithm for pipes, pumps and PRVs. 
    """            

    """ For pipes. """
    q_relu = torch.relu(q)
    q_x = torch.pow(q_relu, 1.852)
    l = q_x * r

    """ For pumps. """
    if (m_e_pump == 1).sum() != 0:
        pump_ccs_orig = pump_ccs.clone()
        pump_ccs = pump_ccs_orig[m_e_pump[:,0] == 1, :]
        l_pumps = (-1 * pump_ccs[:, 3:4]**2 * (pump_ccs[:, 0:1] - pump_ccs[:, 1:2] * (q_relu[m_e_pump[:,0] == 1, :] / pump_ccs[:, 3:4])**pump_ccs[:, 2:3])) 
   
        l[m_e_pump[:,0] == 1, :] = l_pumps

    J = 0
    h_updated = torch.zeros_like(h)
    h_star = h.clone()

    while torch.equal(h, h_updated) is False:
        h_updated = h.clone()

        sndr_node_attr = h[edge_index[0, :], :]
        msg = sndr_node_attr - l

        """ For pumps. """
        msg[m_e_pump == -1] = 0
        msg[m_e_pump == 2] = 0
        msg[m_e_pump == -2] = 0

        h_max = scatter(msg, 
                        dim=0, 
                        index=edge_index[1, :], 
                        reduce='max', 
                        out=torch.zeros_like(h)
                        )
        h = torch.maximum(h, h_max)  

        """ Additional update for PRVs. """
        h = torch.minimum(h, m_n_prv)  

        h[reservoir_mask] = h_star[reservoir_mask]
        J += 1      

    return h, q_x, J




def normalize_hydraulics(heads, m_n_prv, pump_curve_ccs, flows, rs, demands,  
                         rmax_offset=1., dmax_offset=1.):
    """ 
        Method to apply our unique physics-preserving normalization. 
    """

    """ Computing the sum of demands. """
    d_max = demands.sum(dim=1).unsqueeze(1) 
    d_max = d_max / dmax_offset
    d_max_nodes = d_max.repeat(1, demands.shape[1], 1) 
    d_max_edges = d_max.repeat(1, rs.shape[1], 1) 

    """ Computing the normalization factor for r. """
    r_max = (3*rs.std(dim=1)).unsqueeze(1) / rmax_offset
    r_max_nodes = r_max.repeat(1, demands.shape[1], 1)
    r_max_edges = r_max.repeat(1, rs.shape[1], 1)

    """ Normalizing all features as defined in the paper. """
    demands_nrmlzd = demands / d_max_nodes
    flows_nrmlzd = flows / d_max_edges 
    rs_nrmlzd = rs / r_max_edges
    heads_nrmlzd = heads / (torch.pow(d_max_nodes, 1.852) * r_max_nodes)
    m_n_prv_nrmlzd = m_n_prv / (torch.pow(d_max_nodes, 1.852) * r_max_nodes)
    pump_curve_ccs_nrmlzd = pump_curve_ccs.clone()
    pump_curve_ccs_nrmlzd[...,0:1] = pump_curve_ccs[...,0:1] / (torch.pow(d_max_edges, 1.852) * r_max_edges)
    pump_curve_ccs_nrmlzd[...,1:2] = (pump_curve_ccs[...,1:2] * torch.pow(d_max_edges, pump_curve_ccs[...,2:3])) \
                                        / (torch.pow(d_max_edges, 1.852) * r_max_edges)
    return heads_nrmlzd, m_n_prv_nrmlzd, pump_curve_ccs_nrmlzd, flows_nrmlzd, rs_nrmlzd, \
            demands_nrmlzd, d_max_nodes, d_max_edges, r_max_nodes, r_max_edges

def denormalize_hydraulics(heads, h_tilde, m_n_prv, pump_curve_ccs, flows, flows_hat, flows_tilde, rs, 
                           demands, d_hat, d_tilde, d_max_nodes, d_max_edges, 
                           r_max_nodes, r_max_edges):

    """ 
        Method to reverse our unique physics-preserving normalization. 
    """

    demands_dnrmlzd = demands * d_max_nodes
    d_hat_dnrmlzd = d_hat * d_max_nodes
    d_tilde_dnrmlzd = d_tilde * d_max_nodes
    flows_dnrmlzd = flows * d_max_edges 
    flows_hat_dnrmlzd = flows_hat * d_max_edges 
    flows_tilde_dnrmlzd = flows_tilde * d_max_edges 
    rs_dnrmlzd = rs * r_max_edges
    heads_dnrmlzd = heads * (torch.pow(d_max_nodes, 1.852) * r_max_nodes)
    h_tilde_dnrmlzd = h_tilde * (torch.pow(d_max_nodes, 1.852) * r_max_nodes)

    m_n_prv_dnrmlzd = m_n_prv * (torch.pow(d_max_nodes, 1.852) * r_max_nodes)
    pump_curve_ccs_dnrmlzd = pump_curve_ccs.clone()
    pump_curve_ccs_dnrmlzd[...,0:1] = pump_curve_ccs[...,0:1] * (torch.pow(d_max_edges, 1.852) * r_max_edges)
    pump_curve_ccs_dnrmlzd[...,1:2] = (pump_curve_ccs[...,1:2] / torch.pow(d_max_edges, pump_curve_ccs[...,2:3])) \
                                        * (torch.pow(d_max_edges, 1.852) * r_max_edges)

    return heads_dnrmlzd, h_tilde_dnrmlzd, m_n_prv_dnrmlzd, pump_curve_ccs_dnrmlzd,\
            flows_dnrmlzd, flows_hat_dnrmlzd, flows_tilde_dnrmlzd, \
                rs_dnrmlzd, demands_dnrmlzd, d_hat_dnrmlzd, d_tilde_dnrmlzd


def create_graph(inp_file, path_to_data, div_factor=1, data_df=None):
    """ Reads the WDS and scenarios as graphs.

        It requires a path to the Network Structure file *.inp (inp_file).
          
        It needs a path to the saved simulations *.xlsx file (path_to_data) if data_df is None.

        It returns a "wdn_graph" object consisting of node features (X), node coordinates,
        node indices, edge indices and edge attributes.    
     """

    """ Reading the .inp file using wntr package. """
    wds = wntr.network.WaterNetworkModel(inp_file)

    """ Reading edge attributes."""
    n_edges = len(wds.get_graph().edges())
    edge_names = np.array(wds.link_name_list)
    edges_df = pd.DataFrame(index=edge_names, dtype=float)
    edges_df = pd.concat((edges_df, 
                          wds.query_link_attribute('length'),
                          wds.query_link_attribute('diameter'),
                          wds.query_link_attribute('roughness')), 
                          axis=1).reset_index()
    edges_df.columns = ['names', 'length', 'diameter', 'roughness']

    all_edge_indices = np.zeros((2, n_edges), dtype=int)
  
    """ Reading node indices. """
    nodes_df = pd.DataFrame(index=wds.get_graph().nodes())#.set_index(0)
    node_indices = np.arange(nodes_df.shape[0])

    """ Reading node attributes."""
    n_nodes = len(node_indices)
    nodes_df = pd.concat((nodes_df, 
                          wds.query_node_attribute('elevation'), 
                          wds.query_node_attribute('coordinates')), 
                          axis=1).reset_index()
    nodes_df.columns = ['names', 'elevation', 'coordinates']
    node_coords = nodes_df['coordinates'].values

    """ Saving reservoir indices. """
    node_types = wds.query_node_attribute('node_type').values    
    reservoirs = list(np.where(node_types == 'Reservoir')[0])

    """ Reading edge indices. """
    for idx, (name, link) in enumerate(wds.links()):
        all_edge_indices[0, idx] = nodes_df.loc[nodes_df['names'] == link.start_node_name].index.values
        all_edge_indices[1, idx] = nodes_df.loc[nodes_df['names'] == link.end_node_name].index.values

    if data_df is None:
        """ Reading from the xlsx file. """
        scenario = WaterFutureScenario(path_to_data)        
        X_df = scenario.get_pressures()
    else:
        """ Formatting the provided data_df. """
        tmp_df = data_df[0]
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        X_df = tmp_df.astype("float32")
    resample_idx = range(0, X_df.shape[0], div_factor)   

    """ 
        Initializing a (S x N_n x 5) node tensor having simulator estimated Heads (h_wntr), 
        Original Demands (d_star) and pumps, PRVs, and reservoir masks.
    """

    X = torch.zeros(len(resample_idx), n_nodes, 2, dtype=torch.get_default_dtype())
    X[:, :, 0] = torch.tensor(X_df.values[resample_idx,:n_nodes], dtype=torch.get_default_dtype())

    """ Reading demands. """
    if data_df is None:
        X_df_D = scenario.get_demands()
    else:
        tmp_df = data_df[2]
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])
        tmp_df = tmp_df.set_index("Timestamp")
        X_df_D = tmp_df.astype("float32")
    Demands = torch.tensor(X_df_D.values[resample_idx,:n_nodes], dtype=torch.get_default_dtype()) 
    X[:, :, -1] = Demands
    X = X.nan_to_num(0)

    """ Reading elevations.  """
    elevs = torch.tensor(nodes_df['elevation'].values, dtype=torch.get_default_dtype())
    elevs = torch.nan_to_num(elevs, nan=0, posinf=0, neginf=0) 


    """ Relabeling nodes and edges.  """
    node_indices_orig, edge_indices_orig = relabel_nodes(node_indices, all_edge_indices)    

    """ 
        Initializing a (S x N_e x 10) edge tensor having edge features r, edge direction and pumps and PRVs masks,
        pump curve coefficients, and simulator flows for comparison only.
    """
    edge_attr_orig = torch.zeros(X.shape[0], edge_indices_orig.shape[1], 1, dtype=torch.get_default_dtype())

    """ Computing r. """
    ldc = torch.tensor(edges_df[['length', 'diameter', 'roughness']].values, dtype=torch.get_default_dtype())
    constnt = 10.667 
    r = (constnt * ldc[..., 0:1]) * \
        (torch.pow(ldc[..., 2:3], -1.852) * \
        torch.pow(ldc[..., 1:2], -4.871))
    r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0) 
    edge_attr_orig[:, :, 0:1] = r

    """     
        PRVS and Pumps. 
    """
    """     
        PRVs.
    """    
    X_ppf = torch.zeros(len(resample_idx), n_nodes, 2, dtype=torch.get_default_dtype())
    edge_attr_ppf = torch.zeros(X.shape[0], edge_indices_orig.shape[1] * 2, 7, dtype=torch.get_default_dtype())    

    valves_df = wds.query_link_attribute('link_type')
    valves_idx = valves_df.str.startswith('Valve').values

    prv_values = wds.query_link_attribute('setting')
    prv_sub_idx = wds.query_link_attribute('valve_type').astype(str).str.startswith('PRV')
    prv_idx = copy.deepcopy(valves_idx)
    prv_idx[valves_idx] = valves_idx[valves_idx] & prv_sub_idx.values
    prv_nodes = edge_indices_orig[:, prv_idx]
    prv_values = torch.tensor(prv_values[prv_idx].values.astype(np.float32), dtype=torch.get_default_dtype())

    prv_values_mask = torch.ones((node_indices_orig.shape[0]), dtype=torch.get_default_dtype()) * X[..., 0].max() * 10      
    prv_values_mask[prv_nodes[1,:]] = prv_values + elevs[prv_nodes[1,:]]

    prv_idx = torch.cat([torch.tensor(prv_idx), torch.tensor(prv_idx) * -1], dim=-1)

    X_ppf[:, :, 0] = prv_values_mask
    edge_attr_ppf[:, :, 0] = prv_idx

    """     
        Pumps 
    """
    pump_start_nodes, pump_nodes, pump_curves_indxs = [], [], []
    pump_idx = np.zeros((len(wds.link_name_list), 5), dtype=np.float32)
    for idx, pump in enumerate(wds.pump_name_list): 
        pump_edge = np.array([ p==pump for p in wds.link_name_list ], dtype=np.float32)
        if wds.links._data[pump].status == 1:
            pump_idx[:, 0] += pump_edge

            pump_start_node = edge_indices_orig[0, pump_edge != 0]
            pump_start_nodes.append(pump_start_node.item())
            pump_node = edge_indices_orig[1, pump_edge != 0]
            pump_nodes.append(pump_node.item())

            if wds.links._data[pump].pump_type == 'HEAD':
                A, B, C = wds.links._data[pump].get_head_curve_coefficients()
            elif wds.links._data[pump].pump_type == 'POWER':
                P = wds.links._data[pump]._base_power / 1000
                A, B, C = 1e-24, -1*P/9.80665, -1

            W = wds.links._data[pump].base_speed
            pump_idx[pump_edge == 1, 1:5] = torch.tensor([A, B, C, W], dtype=torch.get_default_dtype())
            pump_idx[pump_idx[:, 0] == 0, 1:5] = 1.
            pump_curves_indxs.append(idx + 1)
        else:
            pump_idx[:, 0] += pump_edge * 2

    pump_nodes_mask = torch.zeros((node_indices_orig.shape[0]), dtype=torch.get_default_dtype())        
    pump_nodes_mask[pump_nodes] = torch.tensor(pump_curves_indxs, dtype=torch.get_default_dtype())
    pump_nodes_mask[pump_start_nodes] = torch.tensor(pump_curves_indxs, dtype=torch.get_default_dtype()) * -1

    pump_idx_bi = torch.cat([torch.tensor(pump_idx), torch.tensor(pump_idx)], dim=0)
    pump_idx_bi[pump_idx.shape[0]:, 0:1][pump_idx_bi[pump_idx.shape[0]:, 0:1] != 0] \
        = pump_idx_bi[pump_idx.shape[0]:, 0:1][pump_idx_bi[pump_idx.shape[0]:, 0:1] != 0] * -1

    X_ppf[:, :, 1] = pump_nodes_mask
    edge_attr_ppf[:, :, 1:6] = pump_idx_bi

    """ Updating the node tensor with pumps, PRVs, and reservoir masks. """
    X = torch.cat((X, X_ppf), dim=-1)
    res_mask = torch.zeros(len(resample_idx), n_nodes, 1, dtype=torch.get_default_dtype())
    res_mask[:, reservoirs, :] = 1.
    X = torch.cat((X, res_mask), dim=-1)

    """ Reading flows estimated by the simulator for comparison only. """
    if data_df is None:
        flows_df = scenario.get_flows() 
    else:
        tmp_df = data_df[1]
        tmp_df['Timestamp'] = pd.to_datetime(tmp_df['Timestamp'])#, unit='s')
        tmp_df = tmp_df.set_index("Timestamp")
        flows_df = tmp_df.astype("float32")
    flows = torch.tensor(flows_df.values[resample_idx,:flows_df.shape[1]], dtype=torch.get_default_dtype()).unsqueeze(2)
    flows = torch.cat((flows, flows * -1), dim=1)

    """ Converting directed edge indices and attributes to bidirectional. """
    edge_indices_orig, edge_attr = convert_to_bi_edges(edge_indices_orig, edge_attr_orig.clone())

    edge_direct_mask = torch.ones(X.shape[0], edge_attr_orig.shape[1], 1, dtype=torch.get_default_dtype()) 
    edge_direct_mask_bi = torch.cat([edge_direct_mask, edge_direct_mask * 0], dim=1)

    """ 
        Updating the edge tensor with pumps, PRVs, and edge direction masks,
        pump curve coefficients and simulator flows.
    """
    edge_attr = torch.cat((edge_attr, edge_direct_mask_bi, edge_attr_ppf, flows), dim=-1)

    edge_index = edge_indices_orig.repeat(X.shape[0], 1, 1)
    edge_attr[:, :, 0:1][edge_attr[:, :, 2:3] != 0] = 1.         # for PRVs
    edge_attr[:, :, 0:1][edge_attr[:, :, 3:4] != 0] = 1.         # for PUMPs

    """
        Creating data object with following important attributes: 
        wdn_graph.X:              [heads, demands, prv_mask, pump_mask, reservoir_mask]
        wdn_graph.edge_attr:      [r, edge_direct_mask, prv_mask, pump_mask, \iota, \kappa, \nu, \omega, simulator_flows] 
        wdn_graph.edge_index:     edge indices for the WDS graph.  
    """
    wdn_graph = WDN_Graph(X, node_coords, node_indices, edge_index, edge_indices_orig, edge_attr)

    return wdn_graph, reservoirs


class WDN_Graph():
    def __init__(self, X=None, node_coords=None, node_indices=None, edge_index=None, 
                 edge_indices_orig=None, edge_attr=None, pump_curve_coefs = None, wntr_demands = None):
        super().__init__()   
        self.X = X
        self.node_coords = node_coords
        self.node_indices = node_indices     
        self.edge_index = edge_index
        self.edge_indices_orig = edge_indices_orig
        self.edge_attr = edge_attr
        self.pump_curve_coefs = pump_curve_coefs
        self.wntr_demands = wntr_demands



class WDN_Dataset_IM(InMemoryDataset):
    """ 
        InMemory Dataset Object.

        Creates a list of separate graphs for each sample. 
        Each graph is characterized by:
            a masked node feature matrix:       x
            an unmasked node feature matrix:    y
            an edge indices matrix:             edge_index
            an edge attributes matrix:          edge_attr
    """ 
    def __init__(self, ):
        super().__init__()     
        self.data_list = None   

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        wdn_graph = self.data[idx]
        return wdn_graph

    def load(self, wds, reservoirs, n_nodes, masked=True):
        
        self.data = []
        Y = wds.X.clone()
        
        for idx in self.data_list:   
            mask = torch.zeros((n_nodes, 1)).float() 
            mask[reservoirs] = 1 
            # adj = torch.ones((wds.X.shape[1], wds.X.shape[1]))
            # adj[wds.edge_index[idx][0], wds.edge_index[idx][1]] = wds.edge_attr[idx][:,0]
            wdn_graph = Data(
                            x = wds.X[idx, :, :].clone(),
                            y = Y[idx, :, :].clone(),  
                            edge_attr = wds.edge_attr[idx],
                            edge_index = wds.edge_index[idx],
                            # edge_weight = adj
                            )    
            if masked:
                wdn_graph.x[mask[:,0] == 0, 0] = 0 
            
            self.data.append(wdn_graph)
        return Y



def load_dataset(wds, n_nodes, reservoirs, masked=True):
    """ 
        Creating and loading the dataset.
    """
    dataset = WDN_Dataset_IM()
    dataset.data_list = np.arange(wds.X.shape[0])
    Y = dataset.load(wds, reservoirs, n_nodes, masked)    
    return dataset, Y  
    

def plot_errors(Y, Y_hat, args, save_dir=None, flag="test", plot=True):
    """
        Plots the Absolute Relative Errors for all nodes.
    """

    n_nodes = Y_hat.shape[1]
    norm_abs_errors = (Y_hat - Y).abs() / (Y.abs())
    norm_abs_errors = torch.nan_to_num(norm_abs_errors, nan=0, posinf=0, neginf=0)
    # mean_abs_errors = (Y_hat - Y).abs().mean(dim=0) / (Y.abs().mean(dim=0))
    mean_abs_errors = norm_abs_errors.mean(dim=0)
    mean_abs_errors = torch.nan_to_num(mean_abs_errors, nan=0, posinf=0, neginf=0)
    p_coefs = np.zeros(n_nodes)
    for node in range(n_nodes):
        p_coefs[node] = pearson_coef(Y[:, node], Y_hat[:, node])
    t = np.arange(n_nodes)
    
    if plot:
        plt.figure(figsize=(20, 12))
        plt.scatter(t, mean_abs_errors, label="Absolute Relative Error", color='r')
        plt.title('Absolute relative error for all nodes', size=24)
        plt.xlabel('Nodes', size=16)
        plt.ylabel('Absolute Error', size=16)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.tight_layout()
        plt.grid()

        plt.savefig(save_dir+"/plot_errors_"+args.model+"_"+str(args.I)+"_"+str(args.n_epochs)+"_"+str(datetime.date.today())+"_"+flag+".jpg")
        plt.close()

    return mean_abs_errors, norm_abs_errors, p_coefs


def pearson_coef(y, y_predict):
    """
        Computes the Pearson Correlation Coefficient.
    """
    y_diff = y - y.mean(dim=0)
    y_predict_diff = y_predict - y_predict.mean(dim=0)
    p_coef = (y_diff * y_predict_diff).sum(dim=0) / \
            (torch.sqrt((y_diff ** 2).sum(dim=0)) * torch.sqrt((y_predict_diff ** 2).sum(dim=0)) + 1e-32)
    return p_coef


def plot_graph(inp_file, e_index, args, save_dir="", node_errors=[], node_labels=None, vmin=None, vmax=None, 
               plot=True, with_labels=True, cmap="summer", flag="orig", 
               edge_errors=None, edge_vmin=None, edge_vmax=None, edge_color=None,
               node_font_size=24, edge_font_size=24, arrows=True, node_size=1500, edge_labels=None,
               node_names=False, width=5, arrowsize=20, figsize=(60, 37), savefig=True):
    """
        Plots the WDS with a spectrum of colors indicating the level of error for every node
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    G = nx.DiGraph()
    edge_list = [ (u, v) for u, v in zip(*np.array(e_index)) ]
    G.add_edges_from(edge_list)
    node_list = range(G.number_of_nodes())

    if plot:
        pos = wn.query_node_attribute('coordinates').values#[:-3]
        fig, ax = plt.subplots(figsize=figsize)
        node_color = node_errors #normalize(node_errors)
        if node_labels is None:
            if node_names:
                node_labels = wn.node_name_list
            else:
                node_labels = node_list
        if edge_color is None:
            edge_color = edge_errors # (normalize(edge_errors, a=0.25))#.tolist()
        # else:
        #     edge_color = 'jet'
        edge_cmap = mpl.cm.get_cmap(name='jet')

        # nx.set_edge_attributes(G, dict(zip(edge_list, edge_color)), 'edge_error')
        # edge_color = nx.get_edge_attributes(G, 'edge_error').values()

        nx.draw_networkx(G, 
            pos=pos,
            node_color=list(node_color), #vmin=vmin, vmax=vmax,
            nodelist=node_list,
            labels={ n: l for n, l in zip(node_list, node_labels) },
            cmap=cmap,
            node_size=node_size, 
            ax=ax,
            edgelist=edge_list,
            width=width, 
            linewidths=width,
            edge_color=list(edge_color), 
            # edge_vmin=edge_vmin, 
            # edge_vmax=edge_vmax,  
            edge_cmap=edge_cmap,
            with_labels=with_labels,
            font_size=node_font_size,
            arrows=arrows,
            arrowsize=arrowsize,
            verticalalignment='top'
            )
        if edge_errors is not None:
            if edge_labels is None:
                edge_labels = { (u, v) : e for u, v, e in zip(*np.array(e_index), np.array(edge_errors)) }
            nx.draw_networkx_edge_labels(G,
                pos = pos,
                edge_labels = edge_labels,
                font_size = edge_font_size,
            )

        if savefig:
            plt.savefig(save_dir+"/_graph_"+args.model+"_"+str(args.I)+"_"+str(args.n_epochs)+"_"+str(datetime.date.today())+"_"+flag+".jpg")
        #plt.close()
        return plt

def plot_timeseries(Y, Y_hat, mask, n_samples=480, args={}, save_dir="", scenario="", flag="test", scatter=False):
    plt.figure(figsize=(25,50))
    t = np.arange(Y.shape[0])    
    for node in range(Y.shape[1]):
        plt.subplot(10, 5, node+1)
        plt.plot(t, Y[:, node], label="Ground Truth", color="orange")
        if scatter:
            Y_hat[Y_hat == 0] = np.nan
            plt.scatter(t, Y_hat[:, node], label="Prediction", color="green", s=[0.75 for n in range(len(t))])
        else:
            plt.plot(t, Y_hat[:, node], label="Prediction", color="green")
        plt.xlabel("Time")
        plt.ylabel("Pressure")
        plt.title(str(mask[node]))
        plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+"/node_timeseries_"+scenario+"_"+str(n_samples)+"_"+str(args.wds)+\
                "_"+str(args.I)+"_"+str(args.model)+"_"+str(args.n_epochs)+"_"+str(datetime.date.today())+"_"+flag+".jpg")
    plt.close()




