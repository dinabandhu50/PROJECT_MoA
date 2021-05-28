import os
import optuna
import config
import torch
import pandas as pd
import numpy as np
import utils

DEVICE = "cuda:0"
# DEVICE = "cpu"
EPOCHS = 10

def run_training(fold,params, save_model=False):
    df = pd.read_csv(config.TRAIN_FEATURES)

    df = df.drop(["cp_type","cp_time","cp_dose"],axis=1)

    target_df = pd.read_csv(config.TRAIN_TRAGET_FOLDS)

    feature_columns = df.drop("sig_id",axis=1).columns
    target_columns = target_df.drop(["sig_id","kfold"],axis=1).columns

    df = df.merge(target_df,on="sig_id", how="left")

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    xtrain = train_df[feature_columns].to_numpy()
    ytrain = train_df[target_columns].to_numpy()

    xvalid = valid_df[feature_columns].to_numpy()
    yvalid = valid_df[target_columns].to_numpy()

    train_dataset = utils.MoaDataset(features=xtrain, targets=ytrain)
    valid_dataset = utils.MoaDataset(features=xvalid, targets=yvalid)

    # Do not input the num_workers, as this will cause problems
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=1024, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,batch_size=1024
    )

    model = utils.Model(
        nfeatures=xtrain.shape[1], 
        ntargets=ytrain.shape[1], 
        nlayers=params["num_layers"], 
        hidden_size=params["hidden_size"],
        dropout=params["dropout"]
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["learning_rate"])
    eng = utils.Engine(model, optimizer, device=DEVICE)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0 
    print(f"{'fold'.ljust(5)} {'epoch'.ljust(5)} {'train_loss'.ljust(7)} {'valid_loss'.ljust(7)}")

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        valid_loss = eng.evaluate(valid_loader) 
        print(f"{str(fold).ljust(5)} {str(epoch).ljust(5)} {str(round(train_loss,5)).ljust(10)} {str(round(valid_loss,5)).ljust(10)}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(),os.path.join(config.ROOT_DIR,"models","model_{fold}.bin"))
        else:
            early_stopping_counter += 1 
        
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss

def objective(trial):
    params = {
        "num_layers" : trial.suggest_int("num_layers",1,7),
        "hidden_size": trial.suggest_int("hidden_size",16,2048),
        "dropout": trial.suggest_uniform("dropout",0.1,0.7),
        "learning_rate": trial.suggest_loguniform("learning_rate",1e-6,1e-3)
    }
    all_losses = []
    for f_ in range(5):
        temp_loss = run_training(f_,params, save_model=False)
        all_losses.append(temp_loss)
    
    return np.mean(all_losses)
    
if __name__ == '__main__':
    # run_training(fold=0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("best trials:")
    trial_ = study.best_trial

    print(trial_.values) 
    print(trial_.params) 

    scores = 0.0
    for j in range(5):
        scr = run_training(j,trial_.params, save_model=True)
        scores += scr
    print(scores/5)