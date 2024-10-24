# 这个是AD的那个

import os
import random
import logging as log
import coloredlogs
import cv2
import hydra
import numpy as np
import torch
import yaml

from calculate_metric_binary import calculate_metric_binary_grading, calculate_metric_binary
from dao.itemloader import ItemLoader
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

from tqdm import tqdm
# 这个文件里面的models
from models.createmodels import create_model
from dao.adni.utils import init_transforms, calculate_metric, save_model, calculate_macro_avg_sensitivity
from dao.adni.utils import loading_data, parse_item_progs, calculate_class_weights
# from dao import copy_src
from dao.ece import ECELoss, AdaptiveECELoss, ClasswiseECELoss

coloredlogs.install()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_names = ('grading', 'forecast', 'all', 'forecast_binary', 'grading_binary', 'forecast_multi_class')
task2metrics = {'grading': ['ba', 'ac', 'mauc', 'ba.ka', 'ece', 'eca', 'ada_ece', 'cls_ece', 'ba.eca', 'sen', 'spe'],
                'forecast': ['ba', 'ac', 'mauc', 'ba.eca', 'ece', 'eca', 'ada_ece', 'cls_ece', 'loss', 'sen', 'spe'],
                'forecast_binary': ['ba_ad_cn', 'ba_ad_mci', 'ba_mci_cn', 'ac_ad_cn', 'ac_ad_mci', 'ac_mci_cn', 'mauc_ad_cn',
                              'mauc_ad_mci', 'mauc_mci_cn',
                              'ba.eca', 'ece', 'eca', 'ada_ece', 'cls_ece', 'sen_ad_cn', 'sen_ad_mci',
                              'sen_mci_cn', 'spe_ad_cn', 'spe_ad_mci', 'spe_mci_cn'],
                'grading_binary': ['ba_ad_cn', 'ba_ad_mci', 'ba_mci_cn', 'ac_ad_cn', 'ac_ad_mci', 'ac_mci_cn', 'mauc_ad_cn',
                              'mauc_ad_mci', 'mauc_mci_cn',
                              'ba.eca', 'ece', 'eca', 'ada_ece', 'cls_ece', 'sen_ad_cn', 'sen_ad_mci',
                              'sen_mci_cn', 'spe_ad_cn', 'spe_ad_mci', 'spe_mci_cn'],
                'forecast_multi_class': ['ac_ad', 'ac_mci', 'ac_cn'],

                'all': ['loss']}


save_model = {task: {metric: {'best': float('inf') if 'loss' in metric or 'ece' in metric else -1,
                                "filename": ""} for metric in task2metrics[task]} for task in task_names}

for task in task_names:
    save_model[task] = {}
    for _name in task2metrics[task]:
        if _name == "mse" or "loss" in _name or 'ece' in _name:
            save_model[task][_name] = {'best': 1000000.0, "filename": ""}
        else:
            save_model[task][_name] = {'best': -1, "filename": ""}



# @hydra.main(config_path="configs/config_train.yaml")
@hydra.main(config_path="config", config_name="config_train")
def main(cfg):
    a = 1
    cfg = cfg.config
    cfg.seed = cfg.seed if int(cfg.seed) >= 0 else random.randint(0, 1000000)


    wdir = os.getcwd()
    if not os.path.isabs(cfg.meta_root):
        cfg.meta_root = os.path.join(wdir, cfg.meta_root)
        print(f'cfg.meta_root是{cfg.meta_root}')
   
    if not os.path.isdir(cfg.node):
        os.makedirs(cfg.node, exist_ok=True)

    # print(cfg.pretty())
    # print(OmegaConf.to_yaml(cfg))

    with open("args.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

   
    data_folds = loading_data(cfg, img_root='common/adni/data/image_preprocessed', meta_root='common/adni/data/',
                               meta_filename='adni_fdgpet_prognosis.csv',
                               pkl_meta_filename="cv_split_5folds.pkl", seq_len=cfg.seq_len, seed=cfg.seed)
    df_train, df_val = data_folds[cfg.fold_index - 1]

    print(f'df_train={df_train.describe()}')
    print(f'df_val={df_val.describe()}')
    print(f'Training data:\n{df_train["DXTARGET_0"].value_counts()}')
    print(f'Validation data:\n{df_val["DXTARGET_0"].value_counts()}')

    y0_weights, forecast_weights = calculate_class_weights(df_train, cfg)

    loaders = dict()

    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        # TODO: Undo after debugging
        loaders[f'{stage}'] = ItemLoader(
            meta_data=df, root=cfg.root, batch_size=8, num_workers=cfg.num_workers,
            transform=init_transforms()[stage], parser_kwargs=cfg.parser,
            parse_item_cb=parse_item_progs, shuffle=True if stage == "train" else False, drop_last=False)
    model = create_model(cfg, device, forecast_weights, y0_weights)


    if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
        log.fatal(f'Cannot find pretrained model {cfg.pretrained_model}')
        assert False
    elif cfg.pretrained_model:
        log.info(f'Loading pretrained model {cfg.pretrained_model}')
        try:
            model.load_state_dict(torch.load(cfg.pretrained_model), strict=True)
        except ValueError:
            log.fatal(f'Failed loading {cfg.pretrained_model}')
    template_s = np.load(
        r'E:\Technolgy_learning\Learning_code\AD\common\template\aal90_template_20x24x20.npy')
    template_m = np.load(
        r'E:\Technolgy_learning\Learning_code\AD\common\template\aal90_template_40x48x40.npy')
    templates = [template_s, template_m]
    for epoch_i in range(10):
        # TODO: Debug
        for stage in ["train", "eval"]:
            main_loop(loaders[f'{stage}'], epoch_i, model, cfg, stage, templates)


def whether_update_metrics(batch_i, n_iters):
    return batch_i % 10 == 0 or batch_i >= n_iters - 1


def check_y0_exists(cfg):
    return cfg.diag_coef > 0



def filter_metrics(cfg, metrics):
    filtered_metrics = metrics
    return filtered_metrics


def model_selection(cfg, filtered_metrics, model, epoch_i):
    global save_model
    
    if check_y0_exists(cfg):
        save_model = save_model(
            epoch_i, 'grading', "mauc", filtered_metrics, save_model, model, cfg.node, cond="max",
            mode="scalar")
    
    if cfg.prognosis_coef > 0:
        save_model = save_model(
            epoch_i, 'forecast', "mse", filtered_metrics, save_model, model, cfg.node, cond="min",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        save_model = save_model(
            epoch_i, 'forecast', "ba", filtered_metrics, save_model, model, cfg.node, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        save_model = save_model(
            epoch_i, 'forecast', "mauc", filtered_metrics, save_model, model, cfg.node, cond="max",
            mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        save_model = save_model(
           epoch_i, 'forecast', "ba.eca", filtered_metrics, save_model, model, cfg.node, cond="max",
           mode=f"{cfg.model_selection_mode}_rev" if cfg.model_selection_mode == "beta" else cfg.model_selection_mode)
        save_model = save_model(
            epoch_i, 'forecast', "loss", filtered_metrics, save_model, model, cfg.node, cond="min", mode="scalar")

    save_model = save_model(
        epoch_i, 'all', "loss", filtered_metrics, save_model, model, cfg.node, cond="min", mode="scalar")


def prepare_display_metrics(cfg, display_metrics, middle_metrics ):
    """
    Prepares the metrics for display by adding grading and prognosis metrics 
    to the display_metrics dictionary based on the configuration and the provided metrics.

    Parameters:
    - cfg: The configuration object containing various settings.
    - display_metrics: A dictionary where the displayable metrics will be stored.
    - middle_metrics : A dictionary containing the actual computed metrics, categorized by tasks like grading and forecast.

    Returns:
    - display_metrics: The updated dictionary containing formatted metrics for display.
    """

    # Check if y0 exists and handle grading metrics
    if check_y0_exists(cfg):
        grading_metrics = middle_metrics ['grading']
        # Add grading-related metrics to the display dictionary
        display_metrics[f'{cfg.grading}:ba'] = grading_metrics['ba']
        display_metrics[f'{cfg.grading}:ac'] = grading_metrics['ac']
        display_metrics[f'{cfg.grading}:mauc'] = grading_metrics['mauc']

        # Optionally add ECE-related metrics if the configuration allows it
        if cfg.display_ece:
            display_metrics[f'{cfg.grading}:ece'] = grading_metrics['ece']
            display_metrics[f'{cfg.grading}:ada_ece'] = grading_metrics['ada_ece']
            display_metrics[f'{cfg.grading}:cls_ece'] = grading_metrics['cls_ece']

    # Handle prognosis metrics if prognosis_coef is enabled
    if cfg.prognosis_coef:
        forecast_metrics = middle_metrics ['forecast']

        # Add prognosis metrics, joining multiple values with hyphens
        display_metrics[f'forecast:ba'] = "-".join(
            [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['ba'].values()])
        display_metrics[f'forecast:ac'] = "-".join(
            [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['ac'].values()])
        display_metrics[f'forecast:mauc'] = "-".join(
            [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['mauc'].values()])

        # Optionally add ECE-related metrics if configured to display them
        if cfg.display_ece:
            display_metrics[f'forecast:ece'] = "-".join(
                [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['ece'].values()])
            display_metrics[f'forecast:ada_ece'] = "-".join(
                [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['ada_ece'].values()])
            display_metrics[f'forecast:cls_ece'] = "-".join(
                [f'{value:.03f}' if value is not None else "" for value in forecast_metrics['cls_ece'].values()])

    return display_metrics


def get_masked_IDs(cfg, batch, mask_name, t=None):
    IDs = batch['data']['input']['ID']
    if "classifier" in cfg.method_name and t is not None:
        t = cfg.target_time - 1
    
    if t is None:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, 0]]
    else:
        return [IDs[i] for i in range(len(IDs)) if batch[mask_name][i, t + 1]]


def main_loop(loader, epoch_i, model, cfg, stage, templates):
    """
       Main loop for model training and evaluation.

       Parameters:
       - loader: Data loader that provides batches of input data.
       - epoch_i: The current epoch index.
       - model: The model being trained or evaluated.
       - cfg: Configuration object containing model and training parameters.
       - stage: The stage of the process, either 'train' or 'eval'.
       - templates: Predefined templates for the task.
       """
    # Global variables for tracking the best performance metrics
    global best_bacc, saved_bacc_model_fullname
    global best_f1, saved_f1_model_fullname
    global best_auc, saved_auc_model_fullname
    global best_ap, saved_ap_model_fullname
    global task_names, task2metrics
    global save_model

    # Define ECE loss functions
    ece_criterion = ECELoss(normalized=True).cuda()
    adaece_criterion = AdaptiveECELoss(normalized=True).cuda()
    clsece_criterion = ClasswiseECELoss(normalized=True).cuda()

    # Get the number of iterations (batches)
    n_iters = len(loader)
    print(f"Number of iterations in loader: {n_iters}")

    # Initialize progress bar for tracking loop progress
    progress_bar = tqdm(range(n_iters), total=n_iters, desc=f"{stage}::{epoch_i}")

    # Initialize accumulated metrics dictionary for storing results
    aggregated_metrics  = {'ID': [], 'loss': [], f'l_{cfg.grading}': [], 'loss_forecast': [], 'loss_y0': [], 'diag_loss': [],
                           'forecast': None, cfg.grading: None}

    # Initialize lists to store metrics for each task
    for task in task_names:
        aggregated_metrics [task] = {'ID': [[] for _ in range(cfg.seq_len)],
                                     'softmax_by': [[] for _ in range(cfg.seq_len)],
                                     'prob': [[] for _ in range(cfg.seq_len)],
                                     'pred': [[] for _ in range(cfg.seq_len)],
                                     'label': [[] for _ in range(cfg.seq_len)]}

    # Initialize grading metrics if configured to predict current KL
    if cfg.predict_current_KL:
        aggregated_metrics [cfg.grading] = {'ID': [], 'pred': [], 'label': [], 'softmax': [], 'prob': []}

    # Set model to evaluation or training mode based on the stage
    model.eval() if stage == "eval" else model.train()

    # Initialize final metrics dictionary
    final_metrics = {}

    # Main loop over batches
    for batch_i in progress_bar:
        # Sample a batch of data
        batch = loader.sample(1)[0]

        # Extract patient IDs from the input data
        IDs = batch['data']['input']['ID']
        aggregated_metrics ['ID'].extend(IDs)

        # Prepare input dictionary for model
        input = {in_key: (
            batch['data']['input'][in_key].to(device) if isinstance(batch['data']['input'][in_key], torch.Tensor) else
            batch['data']['input'][in_key])
                 for in_key in batch['data']['input']}

        # Determine batch size from input
        for inp in input.values():
            if isinstance(inp, torch.Tensor):
                batch_size = inp.shape[0]
                break
            elif isinstance(inp, (tuple, list)) and isinstance(inp[0], torch.Tensor):
                batch_size = inp[0].shape[0]
                break

        # Add sequence length information to input
        out_seq_len = cfg.seq_len
        input['label_len'] = torch.tensor([out_seq_len] * batch_size, dtype=torch.int32).to(device)

        # Prepare targets dictionary
        targets = {k: batch[k].to(device) for k in batch if "data" not in k and isinstance(batch[k], torch.Tensor)}

        # Call a function to generate output from the target (e.g., one-hot encoding)
        losses, outputs = model.fit(templates, input=input, target=targets, batch_i=batch_i, n_iters=n_iters,
                                    epoch_i=epoch_i, stage=stage)

        # Accumulate loss metrics for display
        display_metrics = {}
        for loss_name, loss_value in losses.items():
            if loss_value is not None:
                aggregated_metrics [loss_name].append(loss_value)
                display_metrics[loss_name] = f'{np.mean(aggregated_metrics [loss_name]):.03f}'

        # Initialize metrics for each task
        middle_metrics  = {task: {metric: {i: None for i in range(out_seq_len)} if task != 'grading' else None for metric in
                             task2metrics[task]}
                      for task in task_names}

        # Accumulate losses for prognosis and grading tasks
        aggregated_metrics ['loss_forecast'].append(losses['loss_forecast'])
        aggregated_metrics ['loss_y0'].append(losses['loss_y0'])
        aggregated_metrics ['loss'].append(losses['loss'])
        # Iterate over each time step in the sequence
        for t in range(cfg.seq_len):
            for task in task_names:
                if task == "forecast" and task in outputs:
                    # Process predictions and probabilities for the prognosis task
                    labels = outputs[task]['label'][t].flatten()
                    preds = np.argmax(outputs[task]['prob'][t], axis=-1)
                    probs = outputs[task]['prob'][t]

                    # Accumulate probabilities, predictions, and labels for each time step
                    aggregated_metrics [task]['prob'][t].extend(list(probs))
                    IDs_masked = get_masked_IDs(cfg, batch, 'prognosis_mask_DXTARGET', t)
                    if len(IDs_masked) != outputs[task]['prob'][t].shape[0]:
                        print('Unmatched!')
                    aggregated_metrics [task]['ID'][t].extend(IDs_masked)
                    aggregated_metrics [task]['softmax_by'][t].append(outputs[task]['prob'][t])
                    aggregated_metrics [task]['pred'][t].extend(list(preds))
                    aggregated_metrics [task]['label'][t].extend(list(labels.astype(int)))

            # Update metrics at the end of certain iterations
            if whether_update_metrics(batch_i, n_iters):
                middle_metrics ['forecast']['ba'][t] = calculate_metric(balanced_accuracy_score,
                                                             aggregated_metrics ['forecast']['label'][t],
                                                             aggregated_metrics ['forecast']['pred'][t])
                middle_metrics ['forecast']['ac'][t] = calculate_metric(accuracy_score, aggregated_metrics ['forecast']['label'][t],
                                                             aggregated_metrics ['forecast']['pred'][t])
                middle_metrics ['forecast']['mauc'][t] = calculate_metric(roc_auc_score, aggregated_metrics ['forecast']['label'][t],
                                                               aggregated_metrics ['forecast']['prob'][t],
                                                               multi_class=cfg.rocauc_mode)
                middle_metrics ['forecast']['sen'][t] = calculate_macro_avg_sensitivity('sen',
                                                                             aggregated_metrics ['forecast']['label'][t],
                                                                             aggregated_metrics ['forecast']['pred'][t])
                middle_metrics ['forecast']['spe'][t] = calculate_macro_avg_sensitivity('spe',
                                                                             aggregated_metrics ['forecast']['label'][t],
                                                                             aggregated_metrics ['forecast']['pred'][t])

                middle_metrics  = calculate_metric_binary(middle_metrics , aggregated_metrics , cfg, t)

                if len(aggregated_metrics ['forecast']['label'][t]) > 0 and cfg.display_ece:
                    forecast_probs = torch.tensor(np.concatenate(aggregated_metrics ['forecast']['softmax_by'][t], axis=0)).to(
                        device)
                    forecast_labels = torch.tensor(aggregated_metrics ['forecast']['label'][t]).to(device)
                    middle_metrics ['forecast']['ece'][t] = ece_criterion(forecast_probs, forecast_labels, return_tensor=False)
                    middle_metrics ['forecast']['eca'][t] = 1.0 - middle_metrics ['forecast']['ece'][t]
                    middle_metrics ['forecast']['ada_ece'][t] = adaece_criterion(forecast_probs, forecast_labels, return_tensor=False)
                    middle_metrics ['forecast']['cls_ece'][t] = clsece_criterion(forecast_probs, forecast_labels, return_tensor=False)

        # Handle current KL evaluation for the grading task
        if check_y0_exists(cfg) and cfg.grading in outputs and outputs[cfg.grading] is not None and \
                outputs[cfg.grading]['label'] is not None:
            IDs_masked = get_masked_IDs(cfg, batch, f'prognosis_mask_{cfg.grading}')
            aggregated_metrics [cfg.grading]['ID'].extend(IDs_masked)
            aggregated_metrics [cfg.grading]['pred'].extend(list(np.argmax(outputs[cfg.grading]['prob'], axis=-1)))
            aggregated_metrics [cfg.grading]['label'].extend(list(outputs[cfg.grading]['label']))
            aggregated_metrics [cfg.grading]['softmax'].append(outputs[cfg.grading]['prob'])
            aggregated_metrics [cfg.grading]['prob'].extend(list(outputs[cfg.grading]['prob']))

            if whether_update_metrics(batch_i, n_iters):
                middle_metrics ['grading']['ba'] = calculate_metric(balanced_accuracy_score,
                                                               aggregated_metrics [cfg.grading]['label'],
                                                               aggregated_metrics [cfg.grading]['pred'])
                middle_metrics ['grading']['ac'] = calculate_metric(accuracy_score,
                                                               aggregated_metrics [cfg.grading]['label'],
                                                               aggregated_metrics [cfg.grading]['pred'])
                middle_metrics ['grading']['mauc'] = calculate_metric(roc_auc_score,
                                                                 aggregated_metrics [cfg.grading]['label'],
                                                                 aggregated_metrics [cfg.grading]['prob'],
                                                                 multi_class=cfg.rocauc_mode)
                middle_metrics ['grading']['spe'] = calculate_macro_avg_sensitivity('spe', aggregated_metrics [cfg.grading][
                    'label'], aggregated_metrics [cfg.grading]['pred'])
                middle_metrics ['grading']['sen'] = calculate_macro_avg_sensitivity('sen', aggregated_metrics [cfg.grading][
                    'label'], aggregated_metrics [cfg.grading]['pred'])

                middle_metrics  = calculate_metric_binary_grading(middle_metrics , aggregated_metrics , cfg)

                if len(aggregated_metrics [cfg.grading]['label']) > 0 and cfg.display_ece:
                    grading_probs = torch.tensor(
                        np.concatenate(aggregated_metrics [cfg.grading]['softmax'], axis=0)).to(device)
                    grading_labels = torch.tensor(aggregated_metrics [cfg.grading]['label']).to(device)
                    middle_metrics ['grading']['ece'] = ece_criterion(grading_probs, grading_labels, return_tensor=False)
                    middle_metrics ['grading']['eca'] = 1.0 - middle_metrics ['grading']['ece']
                    middle_metrics ['grading']['ada_ece'] = adaece_criterion(grading_probs, grading_labels,
                                                                        return_tensor=False)
                    middle_metrics ['grading']['cls_ece'] = clsece_criterion(grading_probs, grading_labels,
                                                                        return_tensor=False)

        # Update and display metrics periodically
        if whether_update_metrics(batch_i, n_iters):
            display_metrics = prepare_display_metrics(cfg, display_metrics, middle_metrics )
            progress_bar.set_postfix(display_metrics)

        # Final batch handling
        if batch_i >= n_iters - 1:
            final_metrics = middle_metrics 


    # Initialize metrics dictionary for all tasks
    metrics = {'all': {}}
    for task in task_names:
        metrics[task] = {}
        for metric_name in task2metrics[task]:
            if metric_name in final_metrics[task]:
                # Directly assign grading metrics, otherwise assign as a list
                metrics[task][metric_name] = final_metrics[task][metric_name] if task in ['grading', 'grading_binary'] else list(final_metrics[task][metric_name].values())

    # Compute mean loss for prognosis and all tasks
    metrics['forecast']['loss'] = np.mean(aggregated_metrics ['loss_forecast'])
    metrics['all']['loss'] = np.mean(aggregated_metrics ['loss'])

    # Store the model during evaluation stage unless store is skipped
    if stage == "eval" and not cfg.skip_store:
        # Filter or store metrics based on method name
        filtered_metrics = filter_metrics(cfg, metrics) if 'classifier' not in cfg.method_name else metrics
        model_selection(cfg, filtered_metrics, model, epoch_i)

    return metrics, aggregated_metrics 


if __name__ == "__main__":
    main()
