import torch
import numpy as np
from utils.utils import load_dataset
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler 
from tqdm import tqdm
import wandb, os, time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

""" Training """
def train(wds_train, wds_val, model, reservoirs, args, save_dir, out_f):
    """ Initializing hyperparameters. """
    n_epochs, learn_r = args.n_epochs, args.lr

    """ Initiating the Optimizer and Learning rate scheduler. """
    optimizer = Adam(model.parameters(), lr=learn_r, weight_decay=0., eps=1e-12)    
    lr_decay_step, lr_decay_rate = args.decay_step, args.decay_rate
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000, lr_decay_step), gamma=lr_decay_rate)
    
    if args.model_path == None:
        args.model_path = os.path.join(save_dir, "model_"+args.model+"_"+str(args.n_epochs)+"_"+str(args.I)+\
                    "_"+str(args.wds)+".pt")

    """ Checking if training using a partially trained model. """
    if args.warm_start and args.model_path != None:                
        model_state = torch.load(args.model_path)
        model.load_state_dict(model_state["model"])
        optimizer.load_state_dict(model_state["optimizer"])

    n_nodes = wds_train.X.shape[1]
    n_edges = wds_train.edge_attr[0].shape[0]

    """ Loading dataset and creating test and validation splits and batches """
    train_dataset, _ = load_dataset(wds_train, n_nodes, reservoirs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset, _ = load_dataset(wds_val, n_nodes, reservoirs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
   
    """ Train-validation loop """
    for epoch in tqdm(range(n_epochs)): 
   
        train_losses = []
        for batch in train_loader: 
            batch.edge_attr[:, 9:10] = 0.               

            model.train()
            model.zero_grad()
            y_hat = model(batch, r_iter=args.r_iter, epoch=epoch)
            train_loss = model.loss()
            train_loss.backward()
            train_losses.append(train_loss.detach().cpu().item())

            clip_val = 1e-5 
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e+6)
            optimizer.step()         

            if args.wandb:
                wandb.log({ 
                    'training/loss' : train_losses[-1],
                    'learning_rate' : float(opt_scheduler.optimizer.param_groups[0]['lr']),
                    'gradient_norm' : grad_norm,
                    'gradient_norm_after' : grad_norm_after,
                    'epoch' : epoch,
                })          

        
        if args.wandb:
            wandb.log({ 
                'training/loss_mean' : np.mean(train_losses),
                'epoch' : epoch 
            })
        
        opt_scheduler.step()        

        if epoch % (n_epochs//150) == 0:
            model.eval()
            val_losses = []  
            q_loss, d_hat_loss, d_tilde_loss = [], [], []          
            for batch_val in val_loader: 
                batch_val.edge_attr[:, 9:10] = 0.               
                with torch.no_grad():
                    y_hat = model(batch_val, r_iter=args.r_iter, epoch=epoch)
                val_loss = model.loss()
                val_losses.append(val_loss)  
                q_loss.append(model.loss_q)  
                d_hat_loss.append(model.loss_d_hat)  
                d_tilde_loss.append(model.loss_d_tilde)  
                
            mean_val_losses = torch.mean(torch.stack(val_losses)).detach().cpu().item()    

            if args.wandb:
                wandb.log({ 
                    'validation/loss_mean' : torch.mean(torch.stack(val_losses)).detach().cpu().item(),
                    'validation/loss_h' : torch.mean(torch.stack(q_loss)).detach().cpu().item(),
                    'validation/loss_f1' : torch.mean(torch.stack(d_hat_loss)).detach().cpu().item(),
                    'validation/loss_f2' : torch.mean(torch.stack(d_tilde_loss)).detach().cpu().item(),
                    'epoch' : epoch,
                })       

            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(mean_val_losses, 8))
            print("Epoch ", epoch, ": Train loss: ", np.round(np.mean(train_losses), 8), \
                " Val loss: ", np.round(mean_val_losses, 8), file=out_f)

            print("Val losses (last batch) - (d, d_hat): ", np.round(torch.mean(torch.stack(d_hat_loss)).detach().cpu().item(), 8), \
                ", (d, d_tilde): ", np.round(torch.mean(torch.stack(d_tilde_loss)).detach().cpu().item(), 8),
                ", (q, q_tilde): ", np.round(torch.mean(torch.stack(q_loss)).detach().cpu().item(), 8))
            print("Val losses (last batch) - (d, d_hat): ", np.round(torch.mean(torch.stack(d_hat_loss)).detach().cpu().item(), 8), \
                ", (d, d_tilde): ", np.round(torch.mean(torch.stack(d_tilde_loss)).detach().cpu().item(), 8),
                ", (q, q_tilde): ", np.round(torch.mean(torch.stack(q_loss)).detach().cpu().item(), 8), file=out_f)
            
        if epoch % (n_epochs//15) == 0:
            """ Saving the model. """
            state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
            print('model path:', args.model_path)
            print('model path:', args.model_path, file=out_f)
            torch.save(state, args.model_path)

        
    return state, args.model_path

""" Testing """
@torch.no_grad()
def test(wds_test, model, reservoirs, args, save_dir, out_f, e=1e-32, load_model=True):

    """ Loading the trained model. """
    if load_model:
        if args.model_path is None:
            args.model_path = os.path.join(save_dir, "model_"+args.model+"_"+str(args.n_epochs)+"_"+str(args.I)+\
                                "_"+str(args.wds)+".pt")
        model_state = torch.load(args.model_path)
        model.load_state_dict(model_state["model"])

    model.eval()
    
    """ Initializing parameters and loading dataset and batches. """
    n_nodes = wds_test.X.shape[1]
    n_edges = wds_test.edge_attr[0].shape[0]

    test_dataset, Y_test = load_dataset(wds_test, n_nodes, reservoirs)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """" Testing and saving the results. """
    test_losses = []
    Y_hat, F_hat, F_tilde, D_hat, D_tilde = [], [], [], [], []
    t_s = time.time()
    for batch in test_loader:
        batch.edge_attr[:, 9:10] = 0.               

        y_hat = model(batch, r_iter=args.r_iter, zeta=e)
    
        test_loss = model.loss()
        test_losses.append(test_loss.detach().cpu().item())

        Y_hat.append(y_hat)
        F_hat.append(model.q_hat)
        F_tilde.append(model.q_tilde)
        D_hat.append(model.d_hat)
        D_tilde.append(model.d_tilde)

    print("\nEvaluation time is: ", time.time() - t_s, ' seconds. \n')

    Y_hat = torch.stack(torch.vstack(Y_hat).detach().cpu().split(n_nodes))
    F_hat = torch.stack(torch.vstack(F_hat).detach().cpu().split(n_edges))
    F_tilde = torch.stack(torch.vstack(F_tilde).detach().cpu().split(n_edges))
    D_hat = torch.stack(torch.vstack(D_hat).detach().cpu().split(n_nodes))
    D_tilde = torch.stack(torch.vstack(D_tilde).detach().cpu().split(n_nodes))

    print("Test loss: ", np.round(np.mean(test_losses), 8))
    print("Test loss: ", np.round(np.mean(test_losses), 8), file=out_f)
    
    return Y_test, Y_hat, test_losses, F_hat, D_hat, F_tilde, D_tilde

