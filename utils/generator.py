import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import networkx as nx
import wntr
from itertools import product
from tqdm import tqdm
import copy, os, json


def create_rand_wds(n_nodes, max_degree=5):
    print('.........')
    while True:
        try:
            weights = np.random.randint(1, max_degree + 1, size=n_nodes)
            wds_graph = nx.geographical_threshold_graph(n_nodes, theta=int(n_nodes*7), dim=2, pos=None, weight=weights, metric=None, p_dist=None, seed=None)
            g_maxdeg = np.max([j for i,j in wds_graph.degree])
            if nx.is_connected(wds_graph):# and g_maxdeg <= max_degree:
                print(nx.is_connected(wds_graph), g_maxdeg, wds_graph.degree)
                break
            else:
                continue
        except:
            continue

    return wds_graph

def gen_coords(n_nodes, _seed, n_ys=5):
    np.random.seed(_seed)
    n_xs = (n_nodes//n_ys)
    _min, _max = 0.1, 0.9
    x_coords = np.arange(_min, _max, (_max - _min) / n_xs)
    y_coords = np.arange(_min, _max, (_max - _min) / n_ys)
    coords = []
    for x in x_coords:
        for y in y_coords:
            x += np.random.normal(0.05, 0.005, 1)[0]
            y += np.random.normal(0., 0.005, 1)[0]
            coords.append((x, y))
    print(len(coords))
    return coords

def gen_dist_matrix(n_nodes, coords):
    dists = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        coords_i = np.asarray(coords[i])
        for j in range(n_nodes):
            coords_j = np.asarray(coords[j])
            dists[i, j] = np.sqrt(np.square(coords_i[0] - coords_j[0]) + np.square(coords_i[1] - coords_j[1]))
    return dists

def gen_sparse_adj(n_nodes, _seed, dists, max_neighbor=7):
    np.random.seed(_seed)
    Adj = np.zeros((n_nodes, n_nodes))
    for r in range(n_nodes):
        n_neighbors = np.random.randint(3, max_neighbor) 
        sorted, indices = torch.sort(torch.tensor(dists[r, :]), descending=False)
        Adj[r, indices.numpy()[1 : n_neighbors]] = 1
    np.fill_diagonal(Adj, 0)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if Adj[i, j] == 1 and Adj[j, i] == 1:
                Adj[j, i] = 0
    # Adj = np.triu(Adj)
    return Adj

def gen_wds_graph(n_nodes, _seed, max_neighbor=5):
    print('.........')
    
    while True:

        coords = gen_coords(n_nodes, _seed, n_ys=int(np.sqrt(n_nodes)))

        dists = gen_dist_matrix(n_nodes, coords)

        Adj = gen_sparse_adj(n_nodes, _seed, dists, max_neighbor)
    
        edge_indices = np.argwhere(Adj == 1).T
        e_idx = [ (i, j) for i, j in zip(*edge_indices) ]
        wds_graph = nx.Graph()
        wds_graph.add_edges_from(e_idx)

        g_maxdeg = np.max([j for i,j in wds_graph.degree])
        g_avedeg = np.mean([j for i,j in wds_graph.degree])
        print(nx.is_connected(wds_graph), g_maxdeg, g_avedeg, _seed)#, wds_graph.degree)
        if nx.is_connected(wds_graph):
            break
        else:
            _seed = _seed + 2

    return wds_graph, coords, g_maxdeg, g_avedeg, e_idx, _seed

def create_grid_wds(n_nodes, _seed, d_seed, save_dir, max_neighbor=5):

    wds_graph, coords, g_maxdeg, g_avedeg, e_idx, _seed = gen_wds_graph(n_nodes, _seed, max_neighbor=max_neighbor)

    wds = wntr.network.WaterNetworkModel()

    # dem_lo, dem_hi, elev_lo, elev_hi = 0.00001, .0005, .1, .5
    dem_lo, dem_hi, elev_lo, elev_hi = 0.001, .05, .1, .5
    len_lo, len_hi, dia_lo, dia_hi, rough_lo, rough_hi = .1, .2, .35, .55, 1.25, 1.75
    decimal_size = 6

    d_offset, p_offset = .1, .25 
    dlr_offset = 1/30

    """ Actual Seed """
    np.random.seed(seed=_seed)
    print(_seed, np.random.get_state()[1][0])

    elevs = np.random.uniform(elev_lo, elev_hi, size = n_nodes) * 100
    # pattern = np.round(np.random.normal(1, .25, size = 96*7), decimal_size).clip(0) #* 5
    pattern = np.round(np.random.normal(1, .25, size = 48), decimal_size).clip(0) #* 5
    base_demands = np.random.uniform(dem_lo, dem_hi, size = n_nodes)
    print(elevs)

    n_edges = len(e_idx)
    pattern_offset = np.round(np.random.normal(0, p_offset, size = 1), decimal_size) 

    
    """ Offset Seed """
    np.random.seed(d_seed)
    print(d_seed, np.random.get_state()[1][0])

    lengths_offset = np.random.normal(1., dlr_offset, size = n_edges) * 100
    diameters_offset = np.random.normal(0., dlr_offset, size = n_edges)
    roughnesses_offset = np.random.normal(1., dlr_offset, size = n_edges) * 100


    # pattern_offset = np.round(np.random.normal(5, p_offset, size = 1), decimal_size) 

    wds.add_pattern(
        name = "random_week", 
        # pattern = list(np.random.normal(1, 0.33, size = 96*7).clip(0))
        # pattern = list(np.round(np.random.normal(1, 0.1, size = 96*7), decimal_size).clip(0))
        pattern = (pattern + pattern_offset).clip(0)
        # pattern = list(np.random.random(size = 96*7))
        )

    for node in range(n_nodes):
        wds.add_junction(
            name = str(node), 
            base_demand = round(base_demands[node], decimal_size), 
            # base_demand = round(base_demands[node] * (1 + base_demands_offset[node]), decimal_size), 
            demand_pattern = "random_week",  
            elevation = round(elevs[node], decimal_size), 
            coordinates = coords[node], 
            demand_category=None
            )


    """ Actual Seed """
    np.random.seed(_seed)
    print(_seed, np.random.get_state()[1][0])

    for e_i, edge in enumerate(e_idx):
        s_n, e_n = edge
        _len = np.random.uniform(len_lo, len_hi, size = 1)[0] * 100
        _dia = np.random.uniform(dia_lo, dia_hi, size = 1)[0]
        _rgh = np.random.uniform(rough_lo, rough_hi, size = 1)[0] * 100
        wds.add_pipe(
            name = str(e_i),
            start_node_name = str(s_n),
            end_node_name = str(e_n),
            length = round(_len, decimal_size),
            diameter = round(_dia + (diameters_offset[e_i] * _dia), decimal_size),
            roughness = round(_rgh, decimal_size),
            minor_loss = 0.0,
            initial_status = "OPEN",
            check_valve = False,
            )   
        
    wds.add_reservoir(
        name = 'r1', 
        base_head = 100.0, 
        head_pattern = None, 
        coordinates = (0.025, 0.025)
        )

    # wds.add_junction(
    #     name = "50", 
    #     base_demand = np.random.uniform(dem_lo, dem_hi, size = 1)[0], 
    #     demand_pattern = "random_week",  
    #     elevation = np.random.uniform(elev_lo, elev_hi, size = 1)[0] * 100, 
    #     coordinates = (0.05, 0.05), 
    #     demand_category=None
    #     )
    
    # wds.add_valve(
    #     name = str(len(e_idx) + 1),
    #     start_node_name = "50",
    #     end_node_name = "0",
    #     diameter = np.random.uniform(dia_lo, dia_hi, size = 1)[0],
    #     valve_type='PRV',
    #     minor_loss = 0.0,
    #     initial_setting = 20.0,
    #     initial_status = "ACTIVE",
    #     )   

    wds.add_pipe(
        name = str(len(e_idx)),
        start_node_name = "r1",
        end_node_name = "0",
        length = round(np.random.uniform(len_lo, len_hi, size = 1)[0] * 100, decimal_size),
        diameter = round(np.random.uniform(dia_lo, dia_hi, size = 1)[0], decimal_size),
        roughness = round(np.random.uniform(rough_lo, rough_hi, size = 1)[0] * 100, decimal_size),
        minor_loss = 0.0,
        initial_status = "OPEN",
        check_valve = False,
        )   

    wds.options.energy.global_efficiency = 75
    wds.options.energy.global_price = 0
    wds.options.energy.demand_charge = 0

    wds.options.report.status = "Full"
    wds.options.report.summary = "No"

    wds.options.time.duration =  60*60*24*1 # 60*60*24*7 
    wds.options.time.hydraulic_timestep = 60*30 # 60*5 
    wds.options.time.quality_timestep = 60*30 # 60*5 
    wds.options.time.pattern_timestep = 60*30 # 60*5 
    wds.options.time.pattern_start = 0 
    wds.options.time.report_timestep = 60*30 # 60*5 
    wds.options.time.report_start = 0 
    wds.options.time.start_clocktime = 0 
    # wds.options.time.rule_timestep = 60*6
    wds.options.time.statistic = "None" 

    wds.options.hydraulic.inpfile_units = 'CMH' 
    wds.options.hydraulic.trials = 50
    wds.options.hydraulic.accuracy = 0.01
    wds.options.hydraulic.unbalanced = "Continue"
    wds.options.hydraulic.unbalanced_value = 10 

    spec_dict = {
                "n_nodes" : n_nodes,
                "dem_range" : [dem_lo, dem_hi],
                "elev_range" : [elev_lo, elev_hi],
                "length_range" : [len_lo, len_hi],
                "dia_range" : [dia_lo, dia_hi],
                "roughness_range" : [rough_lo, rough_hi],
                "g_maxdeg" : int(g_maxdeg),
                "g_avedeg" : g_avedeg,
                "d_seed" : int(d_seed),
                "_seed" : int(_seed)
                }
    spec_filename = os.path.join(save_dir, "specs.json")
    with open(spec_filename, 'w') as specs_file:
        json.dump(spec_dict, specs_file, indent=4)

    filename = os.path.join(save_dir, "wds.inp")
                # "wds_" + str(n_nodes) + "_" + str(g_maxdeg) + "_" + str(g_avedeg) + "_" + str(_seed) + ".inp")
    out_inp = wntr.epanet.io.InpFile()
    out_inp.write(
            filename = filename, 
            wn = wds,
            units = None, 
            version = 2.2, 
            force_coordinates = False
            )


    return wds, wds_graph, coords, g_maxdeg, g_avedeg, e_idx, _seed, d_seed

if __name__ == '__main__':

    n_nodes = 49 #np.arange(25, 80, 5)
    max_neighbor = 6 #n_nodes // 10    
    orig_seeds = np.array([102, 133, 206, 1125, 716, 2, 339, 786, 73, 420])
    n_graphs = 50
    seeds = copy.deepcopy(orig_seeds)
    for g in range((n_graphs // len(orig_seeds)) - 1):
        seeds = np.concatenate((seeds, orig_seeds+(g+1)*3))
    seeds = list(seeds)
    sx = 0
    _factor = 100
    _seed = seeds[sx]
    d_seeds = np.arange(_seed, _seed + n_graphs, 1)

    for sdx, d_seed in enumerate(d_seeds):
        save_dir = os.path.join("networks", "random", "s" + str(sdx + (sx+1)*_factor))
        if not os.path.isdir(save_dir):
            os.system('mkdir ' + save_dir)

        _seed = seeds[sdx%len(d_seeds)]
        np.random.seed(_seed)
        wds, wds_graph, coords, g_maxdeg, g_avedeg, e_idx, _seed, d_seed = create_grid_wds(n_nodes, _seed, d_seed, save_dir, max_neighbor)
    
        print("WDS " + str(sdx + (sx+1)*_factor) + " generated!")    

        fig, ax = plt.subplots(figsize=(40, 25))
        plot_graph = nx.DiGraph()
        plot_graph.add_edges_from(e_idx)
        nx.draw_networkx(G=plot_graph, pos=coords, node_size=2000, font_size=30, ax=ax, nodelist=range(wds_graph.number_of_nodes()), arrows=True, arrowsize=30)
        plt.savefig(os.path.join(save_dir,
                    "wds_graph" + str(n_nodes) + "_" + str(g_maxdeg) + "_" + str(g_avedeg) + "_" + str(_seed) + '_' + str(d_seed) + ".jpg"))
        plt.close()
