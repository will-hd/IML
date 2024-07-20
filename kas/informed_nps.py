import torch.nn as nn
import torch
import torch.nn.functional as F
from .modules import *
from .utils import MultivariateNormalDiag
from torch.distributions import Bernoulli
import numpy as np
from .attention import MultiheadAttender

class INP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_encoder = XYEncoder(config)
        if config.path in ['deterministic', 'both']:
            self.deter_xy_encoder = XYEncoder(config)
        self.latent_encoder =  ModulatedLatentEncoder(config) # LatentEncoder(config) # 
        self.decoder = Decoder(config)
        self.x_encoder = XEncoder(config)
        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples
        if self.config.data_agg_func == 'cross-attention':
            self.cross_attention = MultiheadAttender(
                config.hidden_dim, config.hidden_dim, config.hidden_dim, n_heads=4
            )

        if config.freeze_meta_networks:
            for param in self.xy_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.x_encoder.parameters():
                param.requires_grad = False
            for name, param in self.latent_encoder.named_parameters():
                if 'knowledge_encoder' not in name: 
                    param.requires_grad = False 
        
    def forward(self, x_context, y_context, x_target, y_target, knowledge=None):
        x_context = self.x_encoder(x_context)  # [bs, num_context, x_transf_dim]
        x_target = self.x_encoder(x_target)  # [bs, num_context, x_transf_dim]

        R, R_deterministic = self.encode_globally(x_context, y_context, x_target)
        z_samples, q_z_Cc, q_zCct, k_deter = self.sample_latent(
            R, x_context, x_target, y_target, knowledge
        )

        # reshape z_samples to the shape of x_target
        R_target = self.target_dependent_representation(R, R_deterministic, k_deter, x_target, z_samples)

        p_yCc = self.decode_target(x_target, R_target)

        # return p_yCc, z_samples, q_z_Cc, q_zCct
        return p_yCc, q_z_Cc, q_zCct
    

    def get_mean_z_pred(self, x_context, y_context, x_target, y_target, knowledge=None):
        x_context = self.x_encoder(x_context)  # [bs, num_context, x_transf_dim]
        x_target = self.x_encoder(x_target)  # [bs, num_context, x_transf_dim]

        R, R_deterministic = self.encode_globally(x_context, y_context, x_target)
        z_samples, q_z_Cc, q_zCct, k_deter = self.sample_latent(
            R, x_context, x_target, y_target, knowledge
        )

        z_samples = q_z_Cc.mean.unsqueeze(0)

        # reshape z_samples to the shape of x_target
        R_target = self.target_dependent_representation(R, R_deterministic, k_deter, x_target, z_samples)

        p_yCc = self.decode_target(x_target, R_target)

        return p_yCc, None, None, None


    def encode_globally(self, x_context, y_context, x_target, knowledge=None):
        """
        Encode context set all together
        """
        Rs = self.xy_encoder(x_context, y_context)

        if self.config.path in ['deterministic', 'both']:
            Rs_deterministic = self.deter_xy_encoder(x_context, y_context)
        else:
            Rs_deterministic = torch.zeros((Rs.shape[0], 1, Rs.shape[-1])).to(Rs.device)

        # aggregate
        if self.config.data_agg_func == 'mean':
            R = torch.mean(Rs, dim=1, keepdim=True)  # [bs, 1, r_dim]
            R_deterministic = torch.mean(Rs_deterministic, dim=1, keepdim=True)
        elif self.config.data_agg_func == 'sum':
            R = torch.sum(Rs, dim=1, keepdim=True)
            R_deterministic = torch.sum(Rs_deterministic, dim=1, keepdim=True)
        elif self.config.data_agg_func == 'none':
            R = Rs
            R_deterministic = Rs_deterministic
        elif self.config.data_agg_func == 'cross-attention':
            Rs = self.cross_attention(x_context, x_target, Rs)
            R = torch.mean(Rs, dim=1, keepdim=True)
            R_deterministic = torch.mean(Rs_deterministic, dim=1, keepdim=True)
    

        if x_context.shape[1] == 0:
            R = torch.zeros((R.shape[0], 1, R.shape[-1])).to(R.device)
            R_deterministic = torch.zeros((R_deterministic.shape[0], 1, R_deterministic.shape[-1])).to(R_deterministic.device)
            

        return R, R_deterministic
    
    def get_knowledge_embedding(self, knowledge):
        return self.latent_encoder.get_knowledge_embedding(knowledge)

    def sample_latent(self, R, x_context, x_target, y_target, knowledge):
        """
        Sample latent variable z given the global representation
        (and during training given the target)
        """
        q_zCc, k_deter = self.infer_latent_dist(R, knowledge, x_context.shape[1])

        if y_target is not None and self.training:
            R_from_target, _ = self.encode_globally(x_target, y_target, x_target)
            q_zCct, _ = self.infer_latent_dist(R_from_target, knowledge, x_target.shape[1])
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])
        # z_samples.shape = [n_z_samples, bs, 1, z_dim]
        return z_samples, q_zCc, q_zCct, k_deter

    def infer_latent_dist(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """
        #knowledge dropout 
        if np.random.uniform() < self.config.knowledge_dropout:
            knowledge = None

        q_z_stats, k_deter = self.latent_encoder(R, knowledge, n)
        q_z_loc, q_z_scale = q_z_stats.split(self.config.hidden_dim, dim=-1)
        q_z_scale = 0.01 + 0.99 * F.softplus(q_z_scale)

        q_zCc = MultivariateNormalDiag(q_z_loc, q_z_scale)

        return q_zCc, k_deter

    def target_dependent_representation(self, R, R_deterministic, k_deter, x_target, z_samples):
        """
        Compute the target dependent representation of the context set
        """
        #R_target = z_samples  # [num_z_samples, batch_size, 1, hidden_dim]
        
        z_samples = z_samples.expand(-1, -1, x_target.shape[1], -1)

        if self.config.path in ['deterministic', 'both']:
            R_deterministic = F.relu(R_deterministic + k_deter)

            # [num_z_samples, batch_size, num_targets, hidden_dim]
            R_deterministic = R_deterministic.unsqueeze(0).expand(z_samples.shape[0], -1, x_target.shape[1], -1)


        if self.config.path == 'deterministic':
            R_target = R_deterministic
        elif self.config.path == 'latent':
            R_target = z_samples
        elif self.config.path == 'both':
            R_target = torch.concat([R_deterministic, z_samples], dim=-1)
        else:
            raise ValueError('Unknown path')

        return R_target

    def decode_target(self, x_target, R_target):
        """
        Decode the target set given the target dependent representation
        """
        if self.config.dataset == 'metaMIMIC':

            logits = self.decoder(x_target, R_target)
            probs = torch.sigmoid(logits)
            probs = probs[:, :, :, 0].unsqueeze(-1)
            p_yCc = Bernoulli(probs=probs)

        else:
            p_y_stats = self.decoder(x_target, R_target)

            p_y_loc, p_y_scale = p_y_stats.split(self.config.output_dim, dim=-1)
            
            # bound the variance (minimum 0.1)
            p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)
            # print('p_y', p_y_loc.shape, p_y_scale.shape)
            p_yCc = MultivariateNormalDiag(p_y_loc, p_y_scale)

        return p_yCc
    

if __name__ == "__main__":
    from argparse import Namespace
    import sys
    import os
    from loss import ELBOLoss

    sys.path.append(os.getcwd())
    from dataset.utils import get_dataloader
    from dataset.datasets import NewRegression
    import numpy as np
    import random

    config = Namespace(
        input_dim=1,
        output_dim=1,
        xy_encoder_num_hidden=2,
        xy_encoder_hidden_dim=128,
        data_agg_func='none',
        latent_encoder_num_hidden=2,
        decoder_hidden_dim=64,
        decoder_num_hidden=2,
        decoder_activation='gelu',
        hidden_dim=128,
        x_transf_dim=128,
        x_encoder_num_hidden=1,
        test_num_z_samples=32,
        train_num_z_samples=1,
        knowledge_extractor_num_hidden=0,
        knowledge_extractor_hidden_dim=128,
        knowledge_dropout=0,
        knowledge_dim=128,
        knowledge_merge='self-attention',
        text_encoder = 'set',
        use_knowledge=True,
        freeze_llm=True,
        tune_llm_layer_norms=False,
        freeze_meta_networks=False,
        # dataset
        batch_size=64,
        min_num_context=0,
        max_num_context=30,
        x_sampler='uniform',
        noise=0,
        # reproducibility
        seed=44,
        dataset='new-regression',
        knowledge_type='set',
        num_targets=50,
        path='latent'
        #xy_self_attention='dot',
    )
    config.device = "cpu"
    
    dataset = NewRegression(root='data/new', split='train')
    #dataset = CelebA(root='data', split='train', knowledge_type='set')
    # dataset = metaMIMIC(config, split='train', root='data/metaMIMIC', knowledge_type='none')
    # train_dataloader = get_dataloader(dataset, config)
    # config.num_classes = dataset.num_classes
    #dataset = SetKnowledgeTrendingSinusoids(split='train', root='data/toy-regression', knowledge_type='masked')
    train_dataloader = get_dataloader(dataset, config)
    #config.knowledge_input_dim = dataset.knowledge_input_dim

    model = INP(config)
    loss_func = ELBOLoss()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
   
    for i, batch in enumerate(train_dataloader):
        context, target, knowledge, _ = batch
        x_context, y_context = context
        x_target, y_target = target

        if config.use_knowledge:
            outputs = model(x_context, y_context, x_target, y_target, knowledge)
        else:
            outputs = model(x_context, y_context, x_target, y_target, None)

        p_yCc = outputs[0]

        loss = loss_func(outputs, y_target)
        print(i, loss)

        # probs = outputs[0].probs
        # predictions = torch.tensor(probs > 0.5, dtype=torch.float32)
        # accuracy = (predictions == y_target).float().mean()

        # print(accuracy)
        
