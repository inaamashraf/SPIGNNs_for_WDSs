import torch
import os, datetime 
import numpy as np
from utils.utils import create_graph, WDN_Graph, normalize_hydraulics, denormalize_hydraulics
from models.models import *
from train_test import train, test
import argparse, json
import wandb
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
                        default = 'train_test',
                        type    = str,
                        choices = ['train_test', 'evaluate'],
                        help    = "train_test i.e. train and test a new model, or evaluate i.e. evaluate on an already trained model; default is train_test. ")
    parser.add_argument('--warm_start',
                        default = False,
                        type    = bool,
                        help    = "specify True if you want to further train a partially trained model. model_path must also be specified; default is False.")
    parser.add_argument('--model_path',
                        default = None,
                        type    = str,
                        help    = "specify model path in case of re-training or evaluation; default is None.")
    parser.add_argument('--model',
                        default = 'SPI_GNN',
                        type    = str,
                        choices = ['PI_GNN', 'SPI_GNN'],
                        help    = "PI_GNN or SPI_GNN; default is SPI_GNN.")   
    parser.add_argument('--start_scenario',
                        default = 1000,
                        type    = int,
                        help    = "specify the start scenario name, must be an integer; default is 1000")
    parser.add_argument('--end_scenario',
                        default = 1999,
                        type    = int,
                        help    = "specify the end scenario name, must be an integer; default is 1999")
    parser.add_argument('--n_samples',
                        default = 6,
                        type    = int,
                        help    = "number of samples per scenario to be used for training; default is 6.")
    parser.add_argument('--batch_size',
                        default = 96,
                        type    = int,
                        help    = "mini-batch size used for training; default is 96.")
    parser.add_argument('--n_epochs',
                        default = 1500,
                        type    = int,
                        help    = "number of epochs of training; default is 1500.")    
    parser.add_argument('--lr',
                        default = 1e-4,
                        type    = float,
                        help    = "learning rate; default is 1e-4.")
    parser.add_argument('--decay_step',
                        default = 150,
                        type    = int,
                        help    = "step of the lr scheduler; default is 150.")
    parser.add_argument('--decay_rate',
                        default = 0.75,
                        type    = float,
                        help    = "decay rate of the lr scheduler; default is 0.75.")
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
    parser.add_argument('--wandb',
                        default = False,
                        type    = bool,
                        help    = "specify True if you want to use Weights and Biases during training; default is False.")
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
    out_f = open(save_dir+"/output_"+args.model+"_"+str(args.n_samples)+"_"+str(datetime.date.today())+".txt", "a")

    """ Reading the WDS dataset across multiple scenarios. """
    X_tvt, edge_index_tvt, edge_attr_tvt = [], [], []
    print('Reading Scenarios ... ')
    for s in range(args.start_scenario, args.end_scenario + 1):
        scenario_path = os.path.join(os.getcwd(),"wds",  args.wds, "toy", "s"+str(s))
        args.inp_file = os.path.join(scenario_path, args.wds+".inp")
        args.path_to_data = os.path.join(scenario_path, "results", "Measurements_All.xlsx")
        wdn_graph, reservoirs = create_graph(args.inp_file, args.path_to_data)
        """ wdn_graph.X:              [heads, demands, prv_mask, pump_mask, reservoir_mask]
            wdn_graph.edge_attr:      [r, edge_direct_mask, prv_mask, pump_mask, \iota, \kappa, \nu, \omega, simulator_flows] 
        """
        
        """ Filtering the dataset based on the specified number of samples. """
        X_s = wdn_graph.X[ : args.n_samples].clone()
        edge_index_s = wdn_graph.edge_index[ : args.n_samples].clone()
        edge_attr_s = wdn_graph.edge_attr[ : args.n_samples].clone()
        X_tvt.append(X_s)
        edge_index_tvt += list(edge_index_s)
        edge_attr_tvt += list(edge_attr_s)

    """ Computing the diameter of the WDS. """
    G = nx.DiGraph()
    edge_list = [ (u, v) for u, v in zip(*np.array(wdn_graph.edge_index[0])) ]
    G.add_edges_from(edge_list)
    wds_dia = nx.diameter(G)
    print('\n ... Done. ')
    print("WDS: ", args.wds)
    print('\nDia of the WDS: ', wds_dia, '\n')
    print("Start, End Scenario: ", args.start_scenario, args.end_scenario)
    print("WDS: ", args.wds, file=out_f)
    print('\nDia of the WDS: ', wds_dia, '\n', file=out_f)
    print("Start, End Scenario: ", args.start_scenario, args.end_scenario, file=out_f)

    X_tvt = torch.vstack(X_tvt)    
    edge_index_tvt = torch.stack(edge_index_tvt)    
    edge_attr_tvt = torch.stack(edge_attr_tvt)  

    """ Applying our physics-preserving normalization. """
    if args.model == "SPI_GNN":
        X_tvt[...,0:1], X_tvt[...,2:3], edge_attr_tvt[...,4:7], edge_attr_tvt[...,9:10], edge_attr_tvt[...,0:1], X_tvt[...,1:2], \
        d_max_nodes, d_max_edges, r_max_nodes, r_max_edges \
        = normalize_hydraulics(X_tvt[...,0:1], X_tvt[...,2:3], edge_attr_tvt[...,4:7], edge_attr_tvt[...,9:10], edge_attr_tvt[...,0:1], X_tvt[...,1:2], \
                                rmax_offset=1e3, dmax_offset=1.)
        X_tvt = torch.cat((X_tvt, d_max_nodes, r_max_nodes), dim=-1)
        edge_attr_tvt = torch.cat((edge_attr_tvt, d_max_edges, r_max_edges), dim=-1)
    
    wds_tvt = WDN_Graph(X=X_tvt, edge_index=edge_index_tvt, edge_attr=edge_attr_tvt)
 
    """ Specifying the model and printing the number of parameters. """
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

    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: ', total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', trainable_params)

    if args.mode == "train_test":

        """ Creating train-val-test splits. """
        n_scenarios = args.end_scenario + 1 - args.start_scenario
        t_samples = n_scenarios * args.n_samples
        train_s, val_s = int(0.6 * t_samples), int(0.8 * t_samples)
        wds_train = WDN_Graph(X=X_tvt[: train_s], edge_index=edge_index_tvt[: train_s], edge_attr=edge_attr_tvt[: train_s])
        wds_val = WDN_Graph(X=X_tvt[train_s : val_s], edge_index=edge_index_tvt[train_s : val_s], edge_attr=edge_attr_tvt[train_s : val_s])
        wds_test = WDN_Graph(X=X_tvt[val_s :], edge_index=edge_index_tvt[val_s :], edge_attr=edge_attr_tvt[val_s :])
        print(wds_train.X.shape, wds_val.X.shape, wds_test.X.shape)    
        print(wds_train.edge_attr.shape, wds_val.edge_attr.shape, wds_test.edge_attr.shape)    

        args_fname = save_dir+"/args_"+args.model+"_"+str(args.n_epochs)+"_"+str(args.n_samples)+"_"+str(datetime.date.today())+".json"
        with open(args_fname, 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        """ Training """
        state, model_path = train(wds_train, wds_val, model, reservoirs, args, save_dir, out_f)

        """ Testing """
        H_star, H_tilde, test_losses, F_hat, D_hat, F_tilde, D_tilde  = test(wds_test, model, reservoirs, args, save_dir, out_f, e=0)
        print("Model: ", args.model_path, file=out_f)

    elif args.mode == 'evaluate':
        wds_test = wds_tvt
        H_star, H_tilde, test_losses, F_hat, D_hat, F_tilde, D_tilde  = test(wds_test, model, reservoirs, args, save_dir, out_f, e=0)
        print("Model: ", args.model_path, file=out_f)

    m_n_prv = wds_test.X[:, :, 2:3]
    D_star = wds_test.X[:, :, 1:2]
    F = wds_test.edge_attr[..., 9:10]
    rs = wds_test.edge_attr[..., 0:1]

    """ Reversing our physics-preserving normalization. """
    if args.model == "SPI_GNN":
        H_star[...,0:1], H_tilde, m_n_prv, wds_test.edge_attr[...,4:7], wds_test.edge_attr[..., 9:10], F_hat, F_tilde, rs, D_star, D_hat, D_tilde = \
            denormalize_hydraulics(H_star[...,0:1], H_tilde, m_n_prv, wds_test.edge_attr[...,4:7], wds_test.edge_attr[..., 9:10], F_hat, F_tilde, rs, D_star, D_hat, D_tilde, 
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

    """ Printing and logging MAEs. """
    print("MAE - (H, H_tilde): ", np.round((H_star_norml - H_tilde_norml).abs().mean().item(), 6))
    print("MAE - (H, H_tilde): ", np.round((H_star_norml - H_tilde_norml).abs().mean().item(), 6), file=out_f)
    print("MAE - (D, D_hat): ", np.round((D_norml - D_hat_norml).abs().mean().item(), 6))
    print("MAE - (D, D_hat): ", np.round((D_norml - D_hat_norml).abs().mean().item(), 6), file=out_f)
    print("MAE - (F, F_hat): ", np.round( (F_norml - F_hat_norml).abs().mean().item(), 6))
    print("MAE - (F, F_hat): ", np.round( (F_norml - F_hat_norml).abs().mean().item(), 6), file=out_f)
    print("MAE - (D, D_tilde): ", np.round((D_norml - D_tilde_norml).abs().mean().item(), 6))
    print("MAE - (D, D_tilde): ", np.round((D_norml - D_tilde_norml).abs().mean().item(), 6), file=out_f)
    print("MAE - (F, F_tilde): ", np.round((F_norml - F_tilde_norml).abs().mean().item(), 6))
    print("MAE - (F, F_tilde): ", np.round((F_norml - F_tilde_norml).abs().mean().item(), 6), file=out_f)



if __name__ == '__main__':
    parser = create_cli_parser()

    args = parser.parse_args()    

    """ Initializing Weights and Biases if specified. """
    if args.wandb and args.train and args.mode != 'evaluate':
        wandb.login()
        wandb.init(project='SPI_GNN', config={
            'start_scenario' : args.start_scenario,
            'end_scenario' : args.end_scenario,
            'model' : args.model,
            'batch_size' : args.batch_size,
            'I' : args.I,
            'latent_dim': args.latent_dim,
            'n_MLP' : args.n_mlp,
            'hp' : [args.n_epochs, args.lr, args.decay_step, args.decay_rate],
            'n_samples': args.args.n_samples,
            'wds' : args.wds
        })

    print(args)
    run(args)

    