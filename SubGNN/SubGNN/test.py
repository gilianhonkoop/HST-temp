import sys
sys.path.insert(0, '..') # add config to path
import config
import train as tr
import os
import json
import random
import numpy as np
import argparse
import wandb
from dotenv import load_dotenv

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run SubGNN")
    parser.add_argument('-task', type=str, default=None, help='Task name (e.g. hpo_metab)')
    parser.add_argument('-tb_name', type=str, default="sg", help='Base Model Name for Tensorboard Log')
    parser.add_argument('-restoreModelPath', type=str, default=None, help='Parent directory of model, hparams, kwargs')
    parser.add_argument('-restoreModelName', type=str, default=None, help='Checkpoint filename to restore')
    parser.add_argument('-data_dir', type=str, default=None, help='Data directory (for AML)')
    parser.add_argument('-wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('-wandb_run_id', type=str, default=None, help='W&B run id to resume and log into')
    parser.add_argument("-max_epochs", type=int, default=200, help="Max number of epochs to train")
    parser.add_argument("-random_seeds", action="store_true", help="Use random seeds from 0-9. Otherwise use random random seeds")
    parser.add_argument('-tb_dir', default="tensorboard_test", type=str)
    parser.add_argument('-no_train', action="store_true")
    args = parser.parse_args()
    return args

def main(args_script):
    args_to_function = {
        "task" : args_script.task,
        "tb_name" : args_script.tb_name,
        "restoreModelPath" : args_script.restoreModelPath,
        "restoreModelName" : args_script.restoreModelName,
        "max_epochs" : args_script.max_epochs,
        "tb_dir" : args_script.tb_dir,

        ## Defaults
        "checkpoint_k": 1,
        "no_checkpointing" : False, #0 and True or 1 and False
        "tb_logging": True,
        "runTest" :  False, 
        "no_save" : False,
        "print_train_times" : False,
        "monitor_metric":'val_micro_f1',
        "opt_n_trials":None,
        "debug_mode":False,
        "subset_data":False,
        "restoreModelName":args_script.restoreModelName,
        "noTrain":False,
        "log_path":None,
        "seed": None,
        "tb_version": None,
        "runTest": False,
        "graph_path": None,
        "subgraphs_path": None,
        "similarities_path": None,
        "shortest_paths_path": None,
        "degree_sequence_path": None,
        "ego_graph_path": None,
        "embedding_path": None
    }
    args = Namespace(**args_to_function)

    if args_script.data_dir:
        args.task = None
        args.graph_path = os.path.join(args_script.data_dir, "edge_list.txt")
        args.subgraphs_path = os.path.join(args_script.data_dir, "subgraphs.pth")
        args.similarities_path = os.path.join(args_script.data_dir, "similarities/")
        args.shortest_paths_path = os.path.join(args_script.data_dir, "shortest_path_matrix.npy")
        args.degree_sequence_path = os.path.join(args_script.data_dir, "degree_sequence.txt")
        args.ego_graph_path = os.path.join(args_script.data_dir, "ego_graphs.txt")
        args.embedding_path = os.path.join(args_script.data_dir, "gin_embeddings.pth")

    wandb_run = None
    if args_script.wandb_project and args_script.wandb_run_id:
        load_dotenv(os.path.join(config.PROJECT_ROOT, ".env"))
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        wandb_run = wandb.init(
            project=args_script.wandb_project,
            id=args_script.wandb_run_id,
            resume="allow",
            name=args_script.tb_name,
        )

    # dict to keep track of results
    exp_results = {
        "test_acc_mean":0, "test_acc_sd":0,"test_micro_f1_mean":0,"test_micro_f1_sd":0,
        "test_auroc_mean":0, "test_auroc_sd":0,
        "test_acc" : [], "test_micro_f1": [], "test_auroc" : [],
        "call" : args_to_function
    }

    # for each seed, train a new model
    for seed in range(10): 
        print(f"Running Round {seed+1}")

        # either use a random seed from 0 to 1000000 or use the default random seeds 0-9
        args.seed = random.randint(0, 1000000) if args_script.random_seeds else seed
        print('Seed used: ', args.seed)
        args.tb_dir = os.path.join(config.PROJECT_ROOT, args.tb_dir)
        args.tb_version = f"version_{seed}"
        if not args_script.no_train: #train the model from scratch
            args.noTrain = False
            args.runTest = True
            test_results = tr.train_model(args)
        else: #read in the model - NOTE that this doesn't differentiaate .ckpt files if multiple are saved
            model_path = os.path.join(config.PROJECT_ROOT,args.tb_dir, args.tb_name, args.tb_version)
            if args.restoreModelName:
                outpath = args.restoreModelName
            else:
                for file in os.listdir(model_path):
                    if file.endswith(".ckpt") and file.startswith("epoch"):
                        outpath = file 
            args.noTrain = True
            args.no_save = True
            args.restoreModelPath = model_path
            args.restoreModelName = outpath
            test_results = tr.train_model(args)

        # keep track of test results for each random seed run
        exp_results['test_micro_f1'].append(float(test_results['test_micro_f1']))
        exp_results['test_acc'].append(float(test_results['test_acc']))
        exp_results['test_auroc'].append(float(test_results['test_auroc']))

        if wandb_run is not None:
            wandb.log({
                'seed': seed,
                'test_micro_f1': float(test_results['test_micro_f1']),
                'test_acc': float(test_results['test_acc']),
                'test_auroc': float(test_results['test_auroc'])
            }, step=seed)
    
    exp_results["test_acc_mean"] = np.mean(exp_results['test_acc'])
    exp_results["test_acc_sd"] = np.std(exp_results['test_acc'])
    exp_results["test_micro_f1_mean"] = np.mean(exp_results['test_micro_f1'])
    exp_results["test_micro_f1_sd"] = np.std(exp_results['test_micro_f1'])
    exp_results["test_auroc_mean"] = np.mean(exp_results['test_auroc'])
    exp_results["test_auroc_sd"] = np.std(exp_results['test_auroc'])
   
    print("OVERALL RESULTS:") # across all random seeds
    print(exp_results)

    # write results for all runs to file
    exp_results_file = open(os.path.join(config.PROJECT_ROOT, args.tb_dir, args.tb_name, "experiment_results.json"),"w")
    exp_results_file.write(json.dumps(exp_results, indent=4))
    exp_results_file.close()

    if wandb_run is not None:
        wandb.log({
            'test_acc_mean': exp_results["test_acc_mean"],
            'test_acc_sd': exp_results["test_acc_sd"],
            'test_micro_f1_mean': exp_results["test_micro_f1_mean"],
            'test_micro_f1_sd': exp_results["test_micro_f1_sd"],
            'test_auroc_mean': exp_results["test_auroc_mean"],
            'test_auroc_sd': exp_results["test_auroc_sd"],
        })
        wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)