

import tensorflow as tf
import optuna
import yaml
import os
from models.tf_blocks import *
from utils.tf_bops import *
from utils.tf_processor import evaluate_deepsets_tf
# from data.BraggnnDataset import *
# from data.DeepsetsDataset import *
from data.tf_DeepsetsDataset import setup_data_loaders_deepsets_tf




def load_configs(task="deepsets", config_dir="examples/"):
    """Load YAML configuration files based on specified task.
    
    Args:
        task (str): Task to load configs for. Either "deepsets" or "braggnn".
        config_dir (str): Directory containing config files.
        
    Returns:
        tuple: (task_configs, search_space) containing model configs and search space for specified task
        
    Raises:
        ValueError: If task is not "deepsets" or "braggnn"
    """
    if task not in ["deepsets", "braggnn"]:
        raise ValueError('Task must be either "deepsets" or "braggnn"')
        
    if task == "deepsets":
        with open(os.path.join(config_dir, "DeepSets/deepsets_search_space.yaml"), "r") as f:
            search_space = yaml.safe_load(f)
        
        with open(os.path.join(config_dir, "DeepSets/deepsets_model_example_configs.yaml"), "r") as f:
            task_configs = yaml.safe_load(f)
            
    else:  # task == "braggnn"
        with open(os.path.join(config_dir, "BraggNN/braggnn_search_space.yaml"), "r") as f:
            search_space = yaml.safe_load(f)
        
        with open(os.path.join(config_dir, "BraggNN/bragg_model_example_configs.yaml"), "r") as f:
            task_configs = yaml.safe_load(f)
    
    return task_configs, search_space


def Deepsets_objective_tf(trial):
    """DeepSets objective using search space config"""
    task_configs, search_space = load_configs(task="deepsets")
    spaces = search_space["search_spaces"]
    hyper_params = search_space["hyperparameters"]
    
    bops = 0
    in_dim, out_dim = 3, 5 #3 kinematic features input, 5 possible particle decay classes

    # Sample architecture parameters
    bottleneck_dim = 2 ** trial.suggest_int("bottleneck_dim", 
                                           *spaces["bottleneck_range"])

    aggregator_type = trial.suggest_categorical("aggregator_type", 
                                              spaces["aggregator_space"])

    if aggregator_type == "mean":
        aggregator = tf.keras.layers.GlobalAveragePooling1D()
    elif aggregator_type == "max":
        aggregator = tf.keras.layers.GlobalMaxPooling1D()
    
    # Initialize networks
    phi_len = trial.suggest_int("phi_len", *hyper_params["phi_len_range"])
    
    phi_layers, phi_acts, phi_norms = sample_MLP_tf(
        trial = trial, 
        in_dim = in_dim, 
        out_dim = bottleneck_dim, 
        prefix ="phi_MLP", 
        search_space=spaces,
        num_layers=phi_len
    )
    phi = tf.keras.Sequential(phi_layers)

    rho_len = trial.suggest_int("rho_len", *hyper_params["rho_len_range"])
    rho_layers, rho_acts, rho_norms = sample_MLP_tf(
        trial, 
        bottleneck_dim, 
        out_dim, 
        "rho_MLP", 
        search_space=spaces,
        num_layers=rho_len
    )
    rho = tf.keras.Sequential(rho_layers)

    model = DeepSetsArchitecture_tf(phi, rho, aggregator)

    print(model)
    print("BOPs:", bops)
    print("Trial ", trial.number, " begins evaluation...")
    
    metrics = evaluate_deepsets_tf(model, train_loader, val_loader, test_loader)
    
    accuracy = metrics['val_accuracy']
    inference_time = metrics['inference_time']
    validation_loss = metrics['val_loss']
    param_count = metrics['param_count']

    with open("./Results/global_search.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Accuracy: {accuracy}, BOPs: {bops}, "
            f"Inference time: {inference_time}, Validation Loss: {validation_loss}, "
            f"Param Count: {param_count}, Hyperparams: {trial.params}\n"
        )
    return accuracy, bops

if __name__ == "__main__":
    
    batch_size = 4096
    num_workers = 4

    os.makedirs("./Results", exist_ok=True)

    deepsets_configs, deepsets_search_space = load_configs(task="deepsets")

    base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"
    
    train_loader, val_loader, test_loader = setup_data_loaders_deepsets_tf(
        base_file_name,
        batch_size=batch_size
    )
    
    study = optuna.create_study(
        sampler=optuna.samplers.NSGAIISampler(population_size=20),
        directions=["maximize", "minimize"]
    )

    # Queue example architectures from config
    study.enqueue_trial(deepsets_configs['base'])
    study.enqueue_trial(deepsets_configs['large'])
    study.enqueue_trial(deepsets_configs['medium'])
    study.enqueue_trial(deepsets_configs['small'])
    study.enqueue_trial(deepsets_configs['tiny'])

    study.optimize(Deepsets_objective_tf, n_trials=5)

