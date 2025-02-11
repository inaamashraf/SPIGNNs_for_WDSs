import torch
import os, datetime, time
import numpy as np
from models.models import *
from utils.utils import create_graph, WDN_Graph, plot_errors, plot_graph, plot_timeseries, normalize_hydraulics, denormalize_hydraulics
from utils.data_generator import *
from train_test import test
import argparse, json
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING=1

def create_cli_parser():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser  = argparse.ArgumentParser()
    parser.add_argument('--wds',
                        default = 'anytown',
                        type    = str,
                        choices = ['anytown', 'hanoi', 'pescara', 'area_c', 'zhijiang', 'modena', 'pa1', 'balerma', 'area_a', 'l_town', 'kl'],
                        help    = "select the WDS; default is anytown. ")
    parser.add_argument('--mode',
                        default = 'demands',
                        type    = str,
                        choices = ['demands', 'diameters'],
                        help    = "evaluate robustness by changing demands or diameters; default is demands. ")
    parser.add_argument('--model_path',
                        default = "trained_models_ijcnn/anytown/model_SPI_GNN_1501_5_anytown.pt",
                        type    = str,
                        help    = "specify model path; default is the trained model for Anytown.")
    parser.add_argument('--model',
                        default = 'SPI_GNN',
                        type    = str,
                        choices = ['PI_GNN', 'SPI_GNN'],
                        help    = "PI_GNN or SPI_GNN; default is SPI_GNN.")   
    parser.add_argument('--batch_size',
                        default = 1000,
                        type    = int,
                        help    = "mini-batch size used for evaluation; default is 1000.")
    parser.add_argument('--I',
                        default = 5,
                        type    = int,
                        help    = "number of GNN layers; default is 5.")
    parser.add_argument('--n_iter',
                        default = 5,
                        type    = int,
                        help    = "minimum number of iterations; default is 5.")
    parser.add_argument('--r_iter',
                        default = 5,
                        type    = int,
                        help    = "maximum number of additional (random) iterations; default is 5.")
    parser.add_argument('--n_mlp',
                        default = 1,
                        type    = int,
                        help    = "number of layers in the MLP; default is 1.")
    parser.add_argument('--M_l',
                        default = 128,
                        type    = int,
                        help    = "latent dimension; default is 128.")
    return parser


def run(args):

    """ Creating directories. """
    file_dir = os.path.dirname(os.path.realpath(__file__)) 
    if not os.path.isdir(os.path.join(file_dir, "tmp")):
        os.system('mkdir ' + os.path.join(file_dir, "tmp"))
    save_dir = os.path.join(file_dir, "tmp", str(datetime.date.today()))
    if not os.path.isdir(save_dir):
        os.system('mkdir ' + save_dir)

    """ Creating an output file to log progress. """
    out_f = open(save_dir+"/output_"+args.model+"_"+str(datetime.date.today())+".txt", "a")

    """ Demand multipliers for every WDS. """
    dm_dict = {
                "anytown":        1.0,
                "hanoi":          0.5,
                "pescara":        0.3,
                "area_c" :        2.0,
                "zhijiang":       0.05,
                "modena":         0.3,
                "balerma":        0.5,
                "pa1":            1.0,
                "marchirural":    3.0,
                "kl":             0.3,
                "area_a":         2.0,
                "c_town":         0.2,
                "l_town":         2.0,
                "pa2":            0.2,
                }

    if args.mode == "demands":
        sigmas = np.arange(.1, 1.1, .1)
    elif args.mode == "diameters":
        sigmas = np.arange(.01, .11, .01)

    """ Initializing settings for data generation and the hydraulic simulator. """
    sim_start_time = '2018-01-01 00:00'
    sim_end_time = '2018-01-21 19:30'
    s = 2000
    pattern_multiplier = 1.
    dem_multiplier = dm_dict[args.wds]
    sigma_d = .01
    mu_dem, sigma_dem = 1., .1
    dem_addition = 0  
    reservoir_multiplier = 1  

    print("\nWDS: ", args.wds)
    print("\nWDS: ", args.wds, file=out_f)

    
    """ Varying the standard deviation of the demands/diameters in 10 steps 10 times. """
    columns=["sigma", "h_mae", "d_hat_mae", "q_hat_mae", "d_tilde_mae", "q_tilde_mae", "wntr_times", "model_times"]
    errors = np.zeros((len(sigmas), len(columns)))
    all_start_time = time.time()
    for j in range(10):
        print("\n j = ", j)
        """ Setting random seed. """
        np.random.seed(np.random.randint(1, 1e6, size=1))
        in_seed = np.random.get_state()[1][0] 

        """ Looping through the range of standard deviation of the demands/diameters. """
        for i in range(len(sigmas)):
            if args.mode == "demands":
                sigma_dem = sigmas[i]
                errors[i, 0] = sigma_dem
            elif args.mode == "diameters":
                sigma_d = sigmas[i]
                errors[i, 0] = sigma_d

            print('\nDia_Std: ', sigma_d)     
            print('Dia_Std: ', sigma_d, file=out_f)     
            print('Demand_Std: ', sigma_dem)     
            print('Demand_Std: ', sigma_dem, file=out_f)     

            """ Generating the dataset and running the hydraulic simulator for comparison later. """
            save_dir = os.path.join(os.getcwd(), "wds", args.wds, "toy")
            scenario_times = []
            scenario_times, heads_df, flows_df, demands_df = \
                run_data_gen(args.wds, s, s, scenario_times, 
                            save_dir, sim_start_time, sim_end_time,
                            pattern_gen=False, pattern_multiplier=pattern_multiplier,
                            sigma_d=sigma_d, mu_dem=mu_dem, sigma_dem=sigma_dem,
                            dem_multiplier=dem_multiplier, dem_addition=dem_addition,
                            reservoir_multiplier=reservoir_multiplier,
                            in_seed=in_seed, save_data=False)    

            """ Reading the WDS dataset. """
            scenario_path = os.path.join(os.getcwd(),"wds",  args.wds, "toy", "s"+str(s))
            args.inp_file = os.path.join(scenario_path, args.wds+".inp")
            args.path_to_data = os.path.join(scenario_path, "results", "Measurements_All.xlsx")
            wdn_graph, reservoirs = create_graph(args.inp_file, args.path_to_data,
                                                        data_df=(heads_df, flows_df, demands_df))
            """ wdn_graph.X:              [heads, demands, prv_mask, pump_mask, reservoir_mask]
                wdn_graph.edge_attr:      [r, edge_direct_mask, prv_mask, pump_mask, \iota, \kappa, \nu, \omega, simulator_flows] 
            """
            X_test = wdn_graph.X.clone()
            edge_index_test = wdn_graph.edge_index.clone()
            edge_attr_test = wdn_graph.edge_attr.clone()
            print("No: of samples: ", X_test.shape[0])
            print("No: of samples: ", X_test.shape[0], file=out_f)

            """ Computing the diameter of the WDS. """
            G = nx.DiGraph()
            edge_list = [ (u, v) for u, v in zip(*np.array(wdn_graph.edge_index[0])) ]
            G.add_edges_from(edge_list)
            wds_dia = nx.diameter(G)

            """ Specifying the model. """
            if args.model == 'PI_GNN': 
                model = PI_GNN( M_n = 2,                    # number of node features (d_star, d_hat).
                                out_dim = 1,                # out dimension is 1 since only flows are directly estimated.
                                M_e = 2,                    # number of edge features (q_hat, q_tilde).
                                M_l = args.M_l,             # specified latent dimension.
                                I = args.I,                 # number of GNN layers.
                                num_layers = args.n_mlp,    # number of NN layers used in every MLP.
                                n_iter = args.n_iter,       # minimum number of iterations.
                                bias = False                # we do not use any bias.
                                ).to(device)
            elif args.model == 'SPI_GNN': 
                model = SPI_GNN( M_n = 3,                   # number of node features (d_star, d_hat, res_mask).
                                out_dim = 1,                # out dimension is 1 since only flows are directly estimated.
                                M_e = 2,                    # number of edge features (q_hat, q_tilde).
                                M_l = args.M_l,             # specified latent dimension.
                                I = args.I,                 # number of GNN layers.
                                dia = wds_dia,              # diameter of the graph.
                                num_layers = args.n_mlp,    # number of NN layers used in every MLP.
                                n_iter = args.n_iter,       # minimum number of iterations.
                                bias = False,               # we do not use any bias.
                                ).to(device)

            """ Applying our physics-preserving normalization. """
            if args.model == "SPI_GNN":
                X_test[...,0:1], X_test[...,2:3], edge_attr_test[...,4:7], edge_attr_test[...,9:10], edge_attr_test[...,0:1], X_test[...,1:2], \
                d_max_nodes, d_max_edges, r_max_nodes, r_max_edges \
                = normalize_hydraulics(X_test[...,0:1], X_test[...,2:3], edge_attr_test[...,4:7], edge_attr_test[...,9:10], edge_attr_test[...,0:1], X_test[...,1:2], \
                                        rmax_offset=1e3, dmax_offset=1.)
                X_test = torch.cat((X_test, d_max_nodes, r_max_nodes), dim=-1)
                edge_attr_test = torch.cat((edge_attr_test, d_max_edges, r_max_edges), dim=-1)

            """ Evaluating on the trained model. """
            wds_test = WDN_Graph(X=X_test, edge_index=edge_index_test, edge_attr=edge_attr_test)
            t_s = time.time()
            H_star, H_tilde, test_losses, F_hat, D_hat, F_tilde, D_tilde  = test(wds_test, model, reservoirs, args, save_dir, out_f, e=0)
            model_time = time.time() - t_s
            print("\nEvaluation time is: ", model_time, ' seconds. \n')            
            print("Model: ", args.model_path)
            print("Model: ", args.model_path, file=out_f)

            m_n_prv = wds_test.X[:, :, 2:3]
            D_star = wds_test.X[:, :, 1:2]
            F = wds_test.edge_attr[..., 9:10]
            rs = wds_test.edge_attr[..., 0:1]

            """ Reversing our physics-preserving normalization. """
            if args.model == "SPI_GNN":
                H_star, H_tilde, m_n_prv, edge_attr_test[...,4:7], F, F_hat, F_tilde, rs, D_star, D_hat, D_tilde = \
                    denormalize_hydraulics(H_star, H_tilde, m_n_prv, edge_attr_test[...,4:7], F, F_hat, F_tilde, rs, D_star, D_hat, D_tilde, 
                                            d_max_nodes=wds_test.X[:, :, -2:-1], d_max_edges=wds_test.edge_attr[..., -2:-1], 
                                            r_max_nodes=wds_test.X[:, :, -1:], r_max_edges=wds_test.edge_attr[..., -1:])

            """ Applying Min-Max Normalization for computing Mean Absolute Error (MAE). """
            H_star_min = H_star[..., 0:1].min(dim=1)[0].repeat(1, H_star.shape[1])
            H_star_max = H_star[..., 0:1].max(dim=1)[0].repeat(1, H_star.shape[1])
            H_star_norml = normalize(H_star[..., 0], _min=H_star_min, _max=H_star_max)
            H_tilde_norml = normalize(H_tilde[..., 0], _min=H_star_min, _max=H_star_max)
            
            D_hat[:, reservoirs, 0] = 0
            D_tilde[:, reservoirs, 0] = 0
            D_hat[D_star == 0] = 0
            D_tilde[D_star == 0] = 0
            D_hat[D_hat < 0] = 0
            D_tilde[D_tilde < 0] = 0

            D_min = D_star.min(dim=1)[0].repeat(1, D_star.shape[1])
            D_max = D_star.max(dim=1)[0].repeat(1, D_star.shape[1])
            D_norml = normalize(D_star[..., 0], _min=D_min, _max=D_max)
            D_hat_norml = normalize(D_hat[..., 0], _min=D_min, _max=D_max)
            D_tilde_norml = normalize(D_tilde[..., 0], _min=D_min, _max=D_max)

            F_min = F.abs().min(dim=1)[0].repeat(1, F.shape[1])
            F_max = F.abs().max(dim=1)[0].repeat(1, F.shape[1])
            F_norml = normalize(F[..., 0], _min=F_min, _max=F_max)
            F_hat_norml = normalize(F_hat[..., 0], _min=F_min, _max=F_max)
            F_tilde_norml = normalize(F_tilde[..., 0], _min=F_min, _max=F_max)

            """ Printing and saving MAEs. """
            print("MAE - (H, H_tilde): ", np.round((H_star_norml - H_tilde_norml).abs().mean().item(), 6))
            print("MAE - (H, H_tilde): ", np.round((H_star_norml - H_tilde_norml).abs().mean().item(), 6), file=out_f)
            print("MAE - (D, D_hat): ", np.round((D_norml - D_hat_norml).abs().mean().item(), 6))
            print("MAE - (D, D_hat): ", np.round((D_norml - D_hat_norml).abs().mean().item(), 6), file=out_f)
            print("MAE - (F, F_hat): ", np.round((F_norml - F_hat_norml).abs().mean().item(), 6))
            print("MAE - (F, F_hat): ", np.round((F_norml - F_hat_norml).abs().mean().item(), 6), file=out_f)
            print("MAE - (D, D_tilde): ", np.round((D_norml - D_tilde_norml).abs().mean().item(), 6))
            print("MAE - (D, D_tilde): ", np.round((D_norml - D_tilde_norml).abs().mean().item(), 6), file=out_f)
            print("MAE - (F, F_tilde): ", np.round((F_norml - F_tilde_norml).abs().mean().item(), 6))
            print("MAE - (F, F_tilde): ", np.round((F_norml - F_tilde_norml).abs().mean().item(), 6), file=out_f)
            errors[i, 1] += np.round((H_star_norml - H_tilde_norml).abs().mean().item(), 6)
            errors[i, 2] += np.round((D_norml - D_hat_norml).abs().mean().item(), 6)
            errors[i, 3] += np.round((F_norml - F_hat_norml).abs().mean().item(), 6)
            errors[i, 4] += np.round((D_norml - D_tilde_norml).abs().mean().item(), 6)
            errors[i, 5] += np.round((F_norml - F_tilde_norml).abs().mean().item(), 6)

            """ Saving simulation and evaluation times of the simulator and the model respectively. """
            errors[i, 6] += np.round(scenario_times[0], 6)
            errors[i, 7] += np.round(model_time, 6)

        """ Saving and diplaying results. """
        errors[:, 1:] = errors[:, 1:]/(j+1)
        errors_df = pd.DataFrame(data=errors, columns=columns)
        print(errors_df.T)
        errors_df.T.to_csv(os.path.join(save_dir, "robustness_"+args.wds+"_"+args.model+"_"+str(sigmas)+".csv"))

    print(errors_df.T, file=out_f)

    print("\nTotal time taken: ", time.time() - all_start_time, " seconds.")



if __name__ == '__main__':
    parser = create_cli_parser()
    args = parser.parse_args()    
    print(args)
    run(args)

        