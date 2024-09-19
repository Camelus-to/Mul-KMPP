import logging
import os
import pickle
import coloredlogs
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from .network import make_network
from common.adni.utils import CATEGORICAL_COLS, NUMERICAL_COLS, CATEGORY2CLASSNUM, IMAGING_COLS, remove_date_from_name
from common.losses import create_loss
from models.roi_features import get_roi_feature
import matplotlib.pyplot as plt

MIN_NUM_PATCHES = 0

coloredlogs.install()


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, key_dim):
        super(CrossModalAttention, self).__init__()
        self.attention_x_to_y = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, kdim=key_dim,
                                                      vdim=key_dim)
        self.attention_y_to_x = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, kdim=key_dim,
                                                      vdim=key_dim)
    def forward(self, x, y):
        attention_output_x_to_y, _ = self.attention_x_to_y(x, y, y)
        attention_output_y_to_x, _ = self.attention_y_to_x(y, x, x)
        combined_attention = torch.add(attention_output_x_to_y, attention_output_y_to_x)
        return combined_attention


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, key_dim):
        super(SelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, kdim=key_dim)

    def forward(self, x):

        attention, _ = self.multihead_attention(x, x, x)

        attention = attention[:, 0, :]

        return attention


class Mul_KMPP(nn.Module):
    def __init__(self, cfg, device, pn_weights=None, y0_weights=None):
        super().__init__()

        self.device = device
        self.cfg = cfg
        self.n_meta_out_features = 0
        self.n_all_features = 0
        self.n_last_img_features = 0
        self.n_metadata = 0
        self.n_patches = 0
        self.n_meta_features = cfg.n_meta_features

        self.input_data = cfg.parser.metadata
        self.aal2_data = cfg.parser.aal2_data
        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None else pn_weights

        if "IMG" in self.input_data:
            self.setup_backbone_network(cfg)

        self.n_all_features += self.n_last_img_features  # 976

        for meta_name in self.input_data:
            meta_name = remove_date_from_name(meta_name)

            if meta_name in NUMERICAL_COLS:
                if meta_name in IMAGING_COLS:
                    setattr(self, f'ft_{meta_name}', self.create_metadata_layers(1, self.n_last_img_features))
                    self.n_patches += 1
                else:
                    setattr(self, f'ft_{meta_name}', self.create_metadata_layers(1, self.n_meta_features))
                    self.n_meta_out_features += self.n_meta_features
                    self.n_metadata += 1

        for meta_name in self.input_data:
            if meta_name in CATEGORICAL_COLS:
                setattr(self, f'ft_{meta_name}',
                        self.create_metadata_layers(CATEGORY2CLASSNUM[meta_name], self.n_meta_features))
                self.n_meta_out_features += self.n_meta_features
                self.n_metadata += 1

        setattr(self, f'ft_aal2',
                self.create_aal2_layers(1, 256))

        for meta_name in self.aal2_data:
            setattr(self, f'ft_{meta_name}',
                    self.create_metadata_layers(1, 512))
            self.n_meta_out_features += self.n_meta_features
            self.n_metadata += 1

        self.n_all_features += self.n_meta_out_features  # n_meta_out_features 是16896

        self.dropout = nn.Dropout(p=cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

        # self.feat_patch_dim = cfg.feat_patch_dim
        self.n_classes = cfg.n_pn_classes

        self.feat_dim = cfg.feat_dim = cfg.feat_dim if isinstance(cfg.feat_dim, int) and cfg.feat_dim > 0 \
            else self.n_meta_features + self.n_last_img_features
        self.setup_transformers_with_imaging(cfg)
        self.setup_roi_with_imaging()
        self.use_tensorboard = False
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
        self.y0_weights = torch.tensor(y0_weights) if y0_weights is not None and cfg.diag_coef > 0 else None
        if pn_weights is not None and cfg.prognosis_coef > 0:
            if self.y0_weights is None:
                self.pn_weights = torch.tensor(pn_weights)
            else:
                self.pn_weights = torch.cat((self.y0_weights.unsqueeze(0), torch.tensor(pn_weights)), 0)
        else:
            self.pn_weights = None
        self.create_balance(128, 976)
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.configure_optimizers()
        self.batch_ind = 0
        self.to(self.device)

    def create_roi(n_input_dim, n_output_dim):
        hidden_dim = 256
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.Linear(n_output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def setup_backbone_network(self, cfg):
        """
        Setup the backbone network for feature extraction based on the configuration.

        Args:
            cfg (Namespace): Configuration object containing model parameters.

        Raises:
            ValueError: If an unsupported backbone or max depth is provided.
        """
        # Validate max depth
        if cfg.max_depth < 1 or cfg.max_depth > 5:
            logging.fatal('Max depth must be in [1, 5].')
            assert False

        self.n_input_imgs = cfg.n_input_imgs

        # Initialize the feature extractor using a network builder function
        self.feature_extractor = make_network(cfg, pretrained=cfg.pretrained,
                                              checkpoint_path=cfg.backbone_checkpoint_path,
                                              input_3x3=cfg.input_3x3, n_channels=cfg.n_channels)

        self.blocks = []

        # Handle different backbone configurations, here for ShuffleNet
        if "shufflenet" in cfg.backbone_name:
            self.blocks.extend(
                [self.feature_extractor.conv1, self.feature_extractor.maxpool, self.feature_extractor.features])

            # Set configurations based on the max depth
            if cfg.max_depth == 5:
                self.blocks.append(self.feature_extractor.conv_last)
                self.n_last_img_ft_size = 3
                if cfg.width_mult < 2:
                    self.n_last_img_features = 1024
                elif cfg.width_mult == 2:
                    self.n_last_img_features = 2048
                else:
                    raise ValueError(f'Unsupported width_mult: {cfg.width_mult}')
            elif cfg.max_depth == 4:
                self.n_last_img_ft_size = 5
                if cfg.width_mult == 1:
                    self.n_last_img_features = 464
                elif cfg.width_mult == 1.5:
                    self.n_last_img_features = 704
                elif cfg.width_mult == 2:
                    self.n_last_img_features = 976
                else:
                    raise ValueError(f'Unsupported width_mult: {cfg.width_mult}')
            else:
                raise ValueError(f'Unsupported max_depth < 4 for shufflenet.')

        else:
            raise ValueError(f'Unsupported backbone: {cfg.backbone_name}')

        # Set image feature projection layer if necessary
        self.n_img_features = cfg.n_img_features
        if self.n_img_features <= 0:
            self.img_ft_projection = nn.Identity()
        else:
            self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
            self.n_last_img_features = cfg.n_img_features

        logging.info(f'[INFO] Number of blocks: {len(self.blocks)}')

        # Adjust the number of patches based on the last feature map size
        self.n_patches += self.n_last_img_ft_size ** 3

    def create_balance(self, n_input_dim, n_output_dim):
        """
        Create a balance network with a series of linear layers and activations.

        Args:
            n_input_dim (int): Input dimension size.
            n_output_dim (int): Output dimension size.

        Returns:
            None: Initializes and assigns a balancing network as self.cc.
        """
        hidden_dim = 256
        self.cc = nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.Linear(n_output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def setup_transformers_with_imaging(self, cfg):
        """
        Setup the transformer models for different stages: diagnosis, prognosis, and ROI.

        Args:
            cfg (Namespace): Configuration object containing model parameters.

        Returns:
            None: Initializes and assigns transformer models for various stages.
        """
        self.seq_len = cfg.seq_len + 1
        self.num_cls_num = cfg.num_cls_num if hasattr(cfg,
                                                      "num_cls_num") and cfg.num_cls_num is not None else self.seq_len

        # Setup feature dimensions for diagnosis
        self.feat_diag_dim = cfg.feat_diag_dim if isinstance(cfg.feat_diag_dim,
                                                             int) and cfg.feat_diag_dim > 0 else self.n_last_img_features

        # Transformer for feature context (metadata + diagnosis)
        self.feat_context = FeaT(
            num_patches=self.n_metadata, with_cls=True, num_cls_num=1, patch_dim=self.n_meta_features,
            num_classes=0, dim=self.n_meta_features, depth=cfg.feat_fusion_depth, heads=cfg

    def setup_roi_with_imaging(self):
        self.roi_feature_extractor = Feature_Extraction(nChannels=16)

    def configure_loss_coefs(self, cfg):
        self.log_vars_y0 = nn.Parameter(torch.zeros((1,))) if self.is_mtl() and self.has_y0() else torch.tensor(0.0)
        self.log_vars_pn = nn.Parameter(torch.zeros((self.seq_len))) if self.is_mtl() and self.has_pn() \
            else torch.zeros((self.seq_len), requires_grad=False)

        # alpha
        self.y0_init_power = cfg.y0_init_power
        self.pn_init_power = cfg.pn_init_power

        if cfg.diag_coef > 0:
            self.alpha_power_y0 = torch.tensor(self.y0_init_power, dtype=torch.float32)
        if cfg.prognosis_coef > 0:
            self.alpha_power_pn = torch.tensor([self.pn_init_power] * self.seq_len, dtype=torch.float32)

        # Show class weights
        if self.pn_weights is not None and self.pn_init_power is not None:
            _pn_weights = self.pn_weights ** self.pn_init_power
        else:
            _pn_weights = None

        # gamma
        if self.is_our_loss(with_focal=True):
            print('Force focal gamma to 1.')
            cfg.focal.gamma = 1.0
        self.gamma_y0 = float(cfg.focal.gamma) \
            if "F" in self.cfg.loss_name and self.has_y0() else None
        self.gamma_pn = np.array([float(cfg.focal.gamma)] * self.seq_len, dtype=float) \
            if "F" in self.cfg.loss_name and self.has_pn() else [None] * self.seq_len

    def configure_crits(self):  # loss_name 是 CE
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   gamma=self.cfg.focal.gamma,
                                   reduction='mean').to(self.device)
        self.crit_diag = create_loss(loss_name=self.cfg.loss_name,
                                     normalized=False,
                                     gamma=self.cfg.focal.gamma,
                                     reduction='mean').to(self.device)

    def get_params(self):
        self.params_main = []
        self.params_extra = []
        for p in self.named_parameters():
            if self.is_log_vars(p[0]) and self.is_mtl():
                self.params_extra.append(p[1])
            else:
                self.params_main.append(p[1])

    def configure_optimizers(self):
        self.get_params()

        self.optimizer = torch.optim.Adam(self.params_main, lr=self.cfg['lr'],
                                          betas=(self.cfg['beta1'], self.cfg['beta2']))

        if self.is_mtl() and self.params_extra:
            self.optimizer_extra = torch.optim.Adam(self.params_extra, lr=self.cfg.extra_optimizer.lr,
                                                    betas=(
                                                        self.cfg.extra_optimizer.beta1, self.cfg.extra_optimizer.beta2))

    def is_log_vars(self, x):
        return "log_vars" in x

    def is_gamma(self, x):
        return "gamma" in x

    def has_y0(self):
        return self.cfg.diag_coef > 0

    def has_pn(self):
        return self.cfg.prognosis_coef > 0

    def is_focal_loss(self):
        return "F" in self.cfg.loss_name

    def is_upper_loss(self):
        return "U" in self.cfg.loss_name

    def is_mtl(self):
        return "MTL" in self.cfg.loss_name

    def is_our_loss(self, with_focal=False):
        if not with_focal:
            return "FMTL" in self.cfg.loss_name or "UMTL" in self.cfg.loss_name
        else:
            return "FMTL" in self.cfg.loss_name

    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy()

    def _compute_probs(self, x, tau=1.0, dim=-1, to_numpy=True):
        tau = tau if tau is not None else 1.0

        probs = torch.softmax(x * tau, dim=dim)

        if to_numpy:
            probs = self._to_numpy(probs)
        return probs

    def forward(self, templates, input, stage, batch_i=None, target=None):
        meta_features = []
        img_measure_features = []
        aal_features = []
        img_features = None
        for input_type in self.input_data:
            if input_type not in input:
                print(f'Input has no {input_type}')
            elif input_type.upper() == "IMG":
                img1 = input[input_type]

                img_features, roi_features = self.forward_img(img1, templates)

            elif input_type == "aal2":
                aal2 = input[input_type]
                for i in range(aal2.shape[1]):
                    element = aal2[:, i:i + 1]
                    _ft = getattr(self, f"ft_{input_type}")(element)
                    _ft = torch.unsqueeze(_ft, 1)
                    _ft = self.dropout_between(_ft)
                    aal_features.append(_ft)


            else:
                _ft = getattr(self, f"ft_{input_type}")(input[input_type])
                # 这个操作就是先把形状的列转成了1， 方便对以后来做乘法操作，比如 _ft是（1*521）形状的
                _ft = input[f'{input_type}_mask'].view(-1, 1) * _ft
                _ft = torch.unsqueeze(_ft, 1)
                _ft = self.dropout_between(_ft)
                if input_type in IMAGING_COLS:
                    img_measure_features.append(_ft)
                else:
                    meta_features.append(_ft)
        mask = input['CDRSB_mask']
        for input_type in self.aal2_data:
            _ft = getattr(self, f"ft_{input_type}")(input[input_type])

            _ft = mask.view(-1, 1) * _ft
            _ft = torch.unsqueeze(_ft, 1)
            _ft = self.dropout_between(_ft)
            meta_features.append(_ft)

        preds, d_attn, f_attn, p_attn, desc_pair = self.apply_feat(img_features, img_measure_features,
                                                                   meta_features, aal_features, roi_features, stage)

        if self.cfg.save_attn:
            self.save_attentions(batch_i, root=self.cfg.log_dir, img=input['IMG'] if "IMG" in self.input_data else None,
                                 metadata_names=self.input_data, diags=preds[:, 0, :],
                                 d_attn=d_attn, f_attn=f_attn, p_attn=p_attn, preds=preds[:, 1:, :], target=target)

        return preds, desc_pair

    def save_attentions(self, batch_i, root, img, metadata_names, d_attn, f_attn, p_attn, preds, target, diags):

        diag_target = target[f'prognosis_{self.cfg.grading}'][:, 0]

        pn_target = target[f'prognosis_{self.cfg.grading}'][:, 1:]

        pn_masks = target[f'prognosis_mask_{self.cfg.grading}'][:, 1:]

        data = {'img': img.to('cpu').detach().numpy(),
                'metadata': metadata_names,
                'D': d_attn.to('cpu').detach().numpy(),
                'F': f_attn.to('cpu').detach().numpy(),
                'P': p_attn.to('cpu').detach().numpy(),
                'preds': preds.to('cpu').detach().numpy(),
                'targets': pn_target.to('cpu').detach().numpy(),
                'mask': pn_masks.to('cpu').detach().numpy(),
                'diags': diags.to('cpu').detach().numpy(),
                'y0': diag_target.to('cpu').detach().numpy()}

        os.makedirs(os.path.join(root, "attn"), exist_ok=True)

        attn_fullname = os.path.join(root, "attn", f"batch_{self.cfg.fold_index}_{batch_i}.pkl")
        print(attn_fullname)

        with open(attn_fullname, 'wb') as f:
            pickle.dump(data, f, 4)

    def apply_feat(self, img_features, img_measure_features, meta_features, aal_features, roi_features, stage):
        """
        Apply features across multiple modalities (image, metadata, AAL, ROI) and
        perform cross-modal attention and feature fusion.

        Args:
            img_features (Tensor): Image features.
            img_measure_features (Tensor): Measurement features (unused here).
            meta_features (list): List of metadata features.
            aal_features (Tensor): AAL features for brain regions.
            roi_features (Tensor): ROI (Region of Interest) features.
            stage (str): Stage of the process ("train" or "eval").

        Returns:
            preds (Tensor): Predicted outputs.
            d_attns, f_attns, p_attns (Tensor): Attention outputs from different stages.
            diag_preds, preds (Tuple[Tensor, Tensor]): Diagnosis and prognosis predictions.
        """

        has_img = img_features is not None
        has_meta = meta_features != []

        if has_img:
            # Rearrange img_features for cross-attention
            img_features = rearrange(img_features, 'b c t h w -> b (t h w) c')

            # Cross-modal attention for combining image and ROI features
            cross_attention_layer = CrossModalAttention(976, 4, 976).to(self.device)
            roi_conv_output = self.cc(roi_features)  # Apply initial transformation to ROI features
            conv1d = nn.Conv1d(in_channels=90, out_channels=125, kernel_size=1).to(self.device)
            roi_transformed = conv1d(roi_conv_output)

            # Apply cross attention between image features and ROI-transformed features
            cross_feature = cross_attention_layer(img_features, roi_transformed)
            final_feature = torch.cat((img_features, roi_transformed, cross_feature), dim=1)

            # Prepare and process AAL features
            aal_features = torch.cat(aal_features, 1).to(self.device)
            _, aal_features, aal_attns = self.feat_roi(aal_features)
            aal_features = aal_features[:, :1, :].repeat(1, final_feature.shape[1], 1)
            aal_features = self.dropout(aal_features)

            # Concatenate AAL features to the final feature if present
            if aal_features.size(0) > 0:
                final_feature = torch.cat((final_feature, aal_features), dim=-1)

        # Apply diagnosis feature processing
        diag_preds, img_descs, d_attns = self.feat_diagnosis(final_feature)

        if has_meta:
            # Concatenate metadata features and process
            meta_features = torch.cat(meta_features, 1)
            _, fusion_features, f_attns = self.feat_context(meta_features)
            meta_features = fusion_features[:, :1, :].repeat(1, img_descs.shape[1], 1)
            meta_features = self.dropout(meta_features)

            # Concatenate diagnosis and metadata features for prognosis prediction
            final_feature = torch.cat((img_descs, meta_features), dim=-1)
            preds, patient_descs, p_attns = self.feat_prognosis(final_feature)

            f_attns = [0, 1]

        return preds, d_attns[-1], f_attns[-1], p_attns[-1], (diag_preds[:, 0, :], preds[:, 0, :])

    def create_metadata_layers(self, n_input_dim, n_output_dim):
        """
        Create metadata processing layers for feature extraction.

        Args:
            n_input_dim (int): Input dimension size.
            n_output_dim (int): Output dimension size.

        Returns:
            nn.Sequential: Sequential layers for metadata processing.
        """
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def create_aal2_layers(self, n_input_dim, n_output_dim):
        """
        Create processing layers for AAL features, involving multiple linear and non-linear transformations.

        Args:
            n_input_dim (int): Input dimension size.
            n_output_dim (int): Output dimension size.

        Returns:
            nn.Sequential: Sequential layers for AAL feature processing.
        """
        hidden_dim = 256
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim),
            nn.ReLU(),
            nn.Linear(n_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output_dim),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def forward_img(self, input, templates):
        """
        Process input images through a series of feature extraction blocks.

        Args:
            input (Tensor): Input image tensor.
            templates (list): Template features for ROI processing.

        Returns:
            features (Tensor): Extracted features.
            roi_features (Tensor): ROI features generated from image data.
        """
        features = []
        if isinstance(input, torch.Tensor):
            input = (input,)

        for x in input:
            for i_b, block in enumerate(self.blocks):
                x = block(x)
                x = self.dropout_between(x) if i_b < len(self.blocks) - 1 else self.dropout(x)

            img_ft = x if self.cfg.feat_use else self.gap(x).view(x.shape[0], -1)
            features.append(img_ft)

        # Concatenate features from different blocks
        features = torch.cat(features, 1)

        # Generate ROI features
        feature_map_s, feature_map_m = self.roi_feature_extractor(input[0])
        feature_map_s = F.interpolate(feature_map_s, size=(20, 24, 20), mode='nearest')
        feature_map_m = F.interpolate(feature_map_m, size=(40, 48, 40), mode='nearest')

        roi_features = get_roi_feature(
            feature_map_s.cpu().detach().numpy(),
            feature_map_m.cpu().detach().numpy(),
            templates[0],
            templates[1]
        )
        roi_features = torch.tensor(roi_features)
        return features, roi_features

    def log_tau(self, batch_i, n_iters, epoch_i, stage):
        """
        Log tau parameters to TensorBoard if required.

        Args:
            batch_i (int): Current batch index.
            n_iters (int): Total number of iterations.
            epoch_i (int): Current epoch index.
            stage (str): Stage of the process ("train" or "eval").
        """
        if batch_i + 1 >= n_iters and "MTL" in self.cfg.loss_name and self.use_tensorboard and stage == "eval":
            taus = {f'pn{i}': self.tau_pn[i] for i in range(self.seq_len)} if self.has_pn() else {}
            self.writer.add_scalars('tau', taus, global_step=epoch_i)

    def apply_constraints_softmax_all(self, s=1.0):
        """
        Apply constraints to all tau variables using softmax or exponential transformation.

        Args:
            s (float): Softmax scaling factor.
        """
        if "U" in self.cfg.loss_name:
            _logits = [1.0 / (torch.exp(self.log_vars_pn) + s)]
            _logits = torch.cat(_logits, 0)
            _softmax = torch.softmax(_logits, dim=0)
            self.tau_pn = _softmax / _softmax.max()
        elif self.is_mtl():
            self.tau_pn = torch.exp(-self.log_vars_pn)
        else:
            self.tau_pn = [1.0] * self.seq_len

    def fit(self, templates, input, target, batch_i, n_iters, epoch_i, stage="train"):
        """
        Function to train or evaluate the model based on the provided stage.

        Args:
            templates (Tensor): Input templates.
            input (Tensor): Input features.
            target (dict): Ground truth labels.
            batch_i (int): Current batch index.
            n_iters (int): Number of iterations.
            epoch_i (int): Current epoch index.
            stage (str): Mode of operation, either "train" or "eval".

        Returns:
            losses (dict): Dictionary of calculated losses.
            outputs (dict): Predictions and labels.
        """

        # Extract the target and mask for prognosis
        pn_target = target[f'prognosis_{self.cfg.grading}']
        pn_masks = target[f'prognosis_mask_{self.cfg.grading}']

        # Forward pass through the model
        preds, desc_pair = self.forward(templates, input, batch_i=batch_i, target=target, stage=stage)

        # Initialize the output dictionary
        outputs = {'pn': {'prob': [], 'label': []}, self.cfg.grading: {'prob': None, 'label': None}}

        # Apply constraints (softmax all)
        self.apply_constraints_softmax_all(s=self.cfg.club.s)

        # Move weights to the appropriate device
        if self.pn_weights is not None:
            self.pn_weights = self.pn_weights.to(self.alpha_power_pn.device)

        # Compute consistency loss
        if desc_pair[0].shape != desc_pair[1].shape:
            print(f'{desc_pair[0].shape} vs {desc_pair[1].shape}')
        cons_loss = torch.sum(F.l1_loss(desc_pair[1], desc_pair[0], reduction='none')) / torch.numel(desc_pair[1])

        # Initialize prognosis losses and a counter
        pn_losses = torch.zeros((self.seq_len), device=self.device)
        n_t_pn = 0

        # Iterate over the sequence length for multi-step prognosis
        for t in range(self.seq_len):
            pn_logits_mask = preds[pn_masks[:, t], t, :]
            pn_target_mask = pn_target[pn_masks[:, t], t]

            # If t > 0, store the probabilities and labels for later steps
            if t > 0:
                outputs['pn']['prob'].append(self._compute_probs(pn_logits_mask, self.tau_pn[t], to_numpy=True))
                outputs['pn']['label'].append(self._to_numpy(pn_target_mask))
            else:
                # Store the predictions for the first step (diagnosis)
                outputs[self.cfg.grading]['prob'] = self._compute_probs(pn_logits_mask, self.tau_pn[t], to_numpy=True)
                outputs[self.cfg.grading]['label'] = self._to_numpy(pn_target_mask).flatten()

                pn_pw_weights = self.pn_weights[0, :] ** self.alpha_power_pn[0] if self.pn_weights is not None else None
                diag_loss = self.crit_pn(desc_pair[0], pn_target_mask, normalized=False, tau=self.tau_pn[t],
                                         alpha=pn_pw_weights, gamma=self.gamma_pn[t])

            # Calculate prognosis loss
            if pn_logits_mask.shape[0] > 0 and pn_target_mask.shape[0] > 0:
                pn_pw_weights = self.pn_weights[t, :] ** self.alpha_power_pn[t] if self.pn_weights is not None else None

                pn_loss = self.crit_pn(pn_logits_mask, pn_target_mask, normalized=False,
                                       tau=self.tau_pn[t], alpha=pn_pw_weights, gamma=self.gamma_pn[t])
                if torch.isnan(pn_loss):
                    print(
                        f'pn_{t} -- tau: {self.tau_pn[t].item()}, gamma: {self.gamma_pn[t].item()}, alpha: {pn_pw_weights}')
                pn_losses[t] = pn_loss
                n_t_pn += 1

        # Prepare loss dictionary
        losses = {}

        # Calculate final loss components
        cur_diag_loss = pn_losses[0]
        prognosis_loss = torch.sum(pn_losses) / n_t_pn if n_t_pn > 0 else torch.tensor(0.0, requires_grad=True)
        total_loss = (self.cfg.prognosis_coef * prognosis_loss +
                      self.cfg.diag_coef * cur_diag_loss +
                      diag_loss + self.cfg.cons_coef * cons_loss)

        # Store losses
        losses['loss_y0'] = cur_diag_loss.item()
        losses['loss_pn'] = prognosis_loss.item()
        losses['loss'] = total_loss.item()
        losses['diag_loss'] = diag_loss.item()

        # If in training mode, update the model
        if stage == "train":
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()

                if self.cfg.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.params_main, self.cfg.clip_norm)

                self.optimizer.step()

        # Handle NaN losses
        if self.is_our_loss() and torch.isnan(total_loss):
            print(f'loss_y0: {cur_diag_loss}, loss_pn: {prognosis_loss}.')

        # If in evaluation mode with multi-task learning, perform backpropagation
        if stage == "eval" and self.is_mtl() and not torch.isnan(total_loss):
            with torch.autograd.set_detect_anomaly(True):
                total_loss.backward()

                if self.cfg.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.params_extra, self.cfg.clip_norm)

                if batch_i + 1 >= n_iters:
                    self.optimizer_extra.step()
                    self.optimizer_extra.zero_grad()

        return losses, outputs


class FeaT(nn.Module):
    def __init__(self, num_patches, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_cls_tokens=1, with_cls=True,
                 n_outputs=1, dropout=0., emb_dropout=0., use_separate_ffn=False):
        super().__init__()
        self.patch_dim = patch_dim  # Patch feature dimension (976)
        self.n_outputs = n_outputs  # Output heads (1 for diagnosis, 6 for prediction)
        self.with_cls = with_cls  # Whether to include classification token (True)
        self.use_separate_ffn = use_separate_ffn  # Use separate feed-forward network (False)

        # Initialize classification token if needed
        if self.with_cls:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls_tokens, dim))

        # Set up position embedding based on the number of outputs or patch dimensions
        if n_outputs == 6:
            self.pos_embedding = nn.Parameter(torch.randn(1, 377, dim))
        elif n_outputs == 1:
            self.pos_embedding = nn.Parameter(torch.randn(1, 376, dim))
        elif patch_dim == 256:
            self.pos_embedding = nn.Parameter(torch.randn(1, 118, dim))  # For ROI transformer
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_tokens, dim))

        # Linear transformation for patch embedding
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer setup
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()  # Identity mapping for the classification token

        # Define MLP head for classification or regression
        self.mlp_heads = nn.ModuleList()
        if self.use_separate_ffn:
            # Create separate MLP for each output if needed
            for _ in range(self.n_outputs):
                self.mlp_heads.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, num_classes)
                ))
        else:
            # Use a shared MLP head for all outputs
            self.mlp_heads.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes)
            ))

    def forward(self, features, mask=None):
        # Convert patches to embeddings and add positional encoding
        x = self.patch_to_embedding(features)

        # Include classification token if enabled
        if self.with_cls:
            cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)  # Expand cls_token to batch size
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x += self.pos_embedding
        x = self.dropout(x)

        # Pass through the transformer
        states, attentions = self.transformer(x, mask)

        # Extract classification token from transformer states
        x = self.to_cls_token(states[:, :self.n_outputs])
        x = self.dropout(x)

        # Apply MLP head(s) for classification/regression
        outputs = []
        for i in range(self.n_outputs):
            if self.use_separate_ffn:
                # Use separate MLP head for each output
                out = self.mlp_heads[i](x[:, i])
            else:
                # Use shared MLP head for all outputs
                out = self.mlp_heads[0](x[:, i])
            outputs.append(out)

        # Stack the outputs along the first dimension if there are multiple outputs
        if outputs:
            outputs = torch.stack(outputs, dim=1)

        # Return final outputs, transformer states, and attention weights
        return outputs, states, attentions


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Self-Attention mechanism that establishes relationships between elements in the input sequence and assigns different attention weights to each element.
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):  # dim: input feature dimension, heads: number of attention heads
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # Scaling factor for attention scores
        # Linear transformations to project the input features into query (Q), key (K), and value (V) spaces.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # Output linear transformation followed by dropout
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):  # x has shape (batch_size, sequence_length, feature_dim)
        b, n, _, h = *x.shape, self.heads  # b: batch_size, n: sequence_length, h: heads
        # Project input to Q, K, and V spaces (shape: (batch_size, sequence_length, 3 * dim))
        qkv = self.to_qkv(x)
        # Rearrange qkv to (batch_size, heads, sequence_length, dim_per_head), split into Q, K, and V
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        # Compute dot-product attention weights
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (batch_size, heads, seq_length, seq_length)

        if mask is not None:
            # Adjust mask to match attention shape and apply it
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        # Apply softmax to get attention probabilities
        attn = dots.softmax(dim=-1)

        # Compute weighted sum of values using the attention probabilities
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # Rearrange back to (batch_size, sequence_length, heads * dim_per_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # Final linear projection
        out = self.to_out(out)

        # Return the output and attention scores (for visualization or further use)
        return out, dots


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim,
                 dropout):  # dim: feature dimension, depth: number of transformer layers
        super().__init__()
        self.layers = nn.ModuleList()  # Use a ModuleList to store multiple layers

        # Define multiple transformer layers with attention and feed-forward networks
        for _ in range(depth):
            # Each transformer layer contains LayerNorm -> Attention -> LayerNorm -> FeedForward
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),  # Layer normalization before attention
                Attention(dim, heads=heads, dropout=dropout),  # Multi-head self-attention
                nn.LayerNorm(dim),  # Layer normalization before feed-forward
                FeedForward(dim, mlp_dim, dropout=dropout)  # Feed-forward network
            ]))

    def forward(self, x, mask=None):  # x: input feature tensor, mask: optional attention mask
        attentions = []  # List to store attention weights from each layer

        # Loop through each layer in the transformer
        for norm1, attn, norm2, ff in self.layers:
            # Apply LayerNorm, Attention, and add residual connection
            o = norm1(x)
            o, attn_weights = attn(o, mask)
            attentions.append(attn_weights)  # Store attention weights
            x = o + x  # Residual connection

            # Apply LayerNorm, FeedForward, and add residual connection
            ff_out = ff(norm2(x))
            x = ff_out + x  # Residual connection

        # Return the final output and the list of attention matrices
        return x, attentions


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(outChannels)
        self.conv2 = nn.Conv3d(outChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class Feature_Extraction(nn.Module):
    def __init__(self, nChannels):
        super(Feature_Extraction, self).__init__()
        # encoder for feature extraction
        self.Conv1 = ConvBlock(1, nChannels)
        self.Conv2 = ConvBlock(nChannels, nChannels * 2)
        self.Conv3 = ConvBlock(nChannels * 2, nChannels * 4)
        self.Conv4 = ConvBlock(nChannels * 4, nChannels * 8)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feature = self.Conv1(x)
        feature = self.AvgPool(feature)
        feature = self.Conv2(feature)
        feature = self.AvgPool(feature)
        out_m = self.Conv3(feature)
        feature = self.AvgPool(out_m)
        out_s = self.Conv4(feature)
        return out_s, out_m
