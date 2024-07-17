import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from attention import DotAttender, MultiheadAttender

class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_hidden, output_size, activation=nn.GELU()
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            (
                [nn.Linear(input_size, hidden_size)]
                + [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)]
                + [nn.Linear(hidden_size, output_size)]
            )
        )
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class XEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_hidden=config.x_encoder_num_hidden,
            output_size=config.x_transf_dim,
        )

    def forward(self, x):
        return self.mlp(x)


class XYEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pairer = MLP(
            input_size=config.x_transf_dim + config.output_dim,
            hidden_size=config.xy_encoder_hidden_dim,
            num_hidden=config.xy_encoder_num_hidden,
            output_size=config.hidden_dim,
        )
        self.config = config
        if self.config.__dict__.get('xy_self_attention') == 'dot':
            print('XY Encoder: Initialising dot product self-attention')
            self.self_attention = DotAttender(
                kq_size=config.hidden_dim,
                value_size=config.hidden_dim,
                out_size=config.hidden_dim,
                is_normalize=True
            )
            self.self_attention_num_layers = config.xy_self_attention_num_layers
        elif self.config.__dict__.get('xy_self_attention') == 'multihead':
            print('XY Encoder: Initialising multihead self-attention')
            self.self_attention = MultiheadAttender(
                kq_size=config.hidden_dim,
                value_size=config.hidden_dim,
                out_size=config.hidden_dim,
                n_heads=4,
            )
            self.self_attention_num_layers = config.xy_self_attention_num_layers
        else:
            self.self_attention = None


    def forward(self, x, y):
        """
        Encode the context set all together
        """
        xy = torch.cat([x, y], dim=-1)
        Rs = self.pairer(xy)
        if self.self_attention is not None:
            for _ in range(self.self_attention_num_layers):
                Rs = self.self_attention(Rs, Rs, Rs)

        return Rs


class RoBERTa(nn.Module):
    def __init__(self, config, return_cls=True):
        super(RoBERTa, self).__init__()

        self.dim_model = 768
        self.llm = RobertaModel.from_pretrained("roberta-base")

        if config.freeze_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        if config.tune_llm_layer_norms:
            for name, param in self.llm.named_parameters():
                if "LayerNorm" in name:
                    param.requires_grad = True

        for name, param in self.llm.named_parameters():
            if name == "pooler.dense.weight" or name == "pooler.dense.bias":
                param.requires_grad = True

        self.device = config.device
        self.tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base', truncation=True, do_lower_case=True
        )

        self.return_cls = return_cls

    def forward(self, knowledge):

        knowledge = self.tokenizer.batch_encode_plus(
            knowledge, return_tensors='pt', 
            return_token_type_ids=True, padding=True, truncation=True
        )

        input_ids = knowledge["input_ids"].to(self.device)
        attention_mask = knowledge["attention_mask"].to(self.device)
        token_type_ids = knowledge["token_type_ids"].to(self.device)

        llm_output = self.llm(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids.squeeze(1),
        )
        #hidden_state = llm_output[0]
        
        if self.return_cls:
            output = llm_output['pooler_output']
        else:
            output = llm_output['last_hidden_state']
        return output


class NoEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_model = config.knowledge_input_dim
        self.device = config.device

    def forward(self, knowledge):
        # check if tensor
        if isinstance(knowledge, torch.Tensor):
            out = knowledge.to(self.device).float()
        else:
            out = torch.stack(knowledge).float().to(self.device)
        # if out.dim() == 2:
        #     out = out.unsqueeze(1)
        return out

class SimpleEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_model = config.num_classes
        self.embedding = nn.Embedding(
            num_embeddings=self.dim_model,
            embedding_dim=self.dim_model,
        )

    def forward(self, knowledge):
        knowledge = torch.tensor(knowledge).long().to(self.embedding.weight.device)
        k = self.embedding(knowledge)
        return k

class SetEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_model = config.knowledge_dim
        self.device = config.device
        self.h1 = MLP(
            input_size=config.knowledge_input_dim,
            hidden_size=config.knowledge_dim,
            num_hidden=1,
            output_size=config.knowledge_dim,
        )
        self.h2 = MLP(
            input_size=config.knowledge_dim,
            hidden_size=config.knowledge_dim,
            num_hidden=1,
            output_size=config.knowledge_dim,
        )
        
    def forward(self, knowledge):
        knowledge = knowledge.to(self.device)
        ks = self.h1(knowledge)
        k = torch.sum(ks, dim=1)
        k = self.h2(k)
        return k    


class KnowledgeEncoder(nn.Module):
    def __init__(self, config):
        super(KnowledgeEncoder, self).__init__()
        if config.text_encoder == 'roberta':
            self.text_encoder = RoBERTa(config)
        elif config.text_encoder == 'simple':
            self.text_encoder = SimpleEmbedding(config)
        elif config.text_encoder == 'none':
            self.text_encoder = NoEmbedding(config)
        elif config.text_encoder == 'set':
            self.text_encoder = SetEmbedding(config)

        if config.knowledge_extractor_num_hidden > 0:
            self.knowledge_extractor = MLP(
                input_size=self.text_encoder.dim_model,
                hidden_size=config.knowledge_extractor_hidden_dim,
                num_hidden=config.knowledge_extractor_num_hidden,
                output_size=config.knowledge_dim,
            )
        else:
            self.knowledge_extractor = nn.Linear(
                self.text_encoder.dim_model, config.knowledge_dim
            )
        self.config = config

    def forward(self, knowledge):
        text_representation = self.text_encoder(knowledge)
        knowledge_representation = self.knowledge_extractor(text_representation)
        return knowledge_representation.unsqueeze(1)


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.knowledge_dim = config.knowledge_dim

        if config.knowledge_merge in ['sum', 'mean', 'self-attention', 'self-attention-2']:
            input_dim = config.hidden_dim

        elif config.knowledge_merge == 'concat':
            input_dim = config.hidden_dim + config.knowledge_dim

        elif config.knowledge_merge == 'mlp':
            input_dim = config.hidden_dim
            self.knowledge_merger = MLP(
                input_size=config.hidden_dim + config.knowledge_dim,
                hidden_size=config.hidden_dim,
                num_hidden=1,
                output_size=config.hidden_dim
            )
        else:
            raise NotImplementedError
            
        if config.use_knowledge:
            self.knowledge_encoder = KnowledgeEncoder(config)
        else:
            self.knowledge_encoder = None

        if config.path in (['both', 'deterministic']):
            self.knowledge_encoder_deterministic = KnowledgeEncoder(config)

        if config.latent_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(
                input_dim, 2 * config.hidden_dim
            )

        if config.knowledge_merge in ['self-attention', 'self-attention-2']:
            self.attention = MultiheadAttender(
                kq_size=config.hidden_dim,
                value_size=config.hidden_dim,
                out_size=config.hidden_dim,
                n_heads=4,
            )

        self.config = config


    def forward(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """
        if knowledge is None:
            k = torch.zeros((R.shape[0], 1, self.knowledge_dim)).to(R.device)
            k_deter = torch.zeros((R.shape[0], 1, self.knowledge_dim)).to(R.device)
        else:
            k = self.knowledge_encoder(knowledge)
            if self.config.path in ['both', 'deterministic']:
                k_deter = self.knowledge_encoder_deterministic(knowledge)
            else:
                k_deter = None

        if self.config.knowledge_merge == 'sum':
            encoder_input = F.relu(R + k)

        elif self.config.knowledge_merge == 'mean':
            encoder_input = torch.mean(torch.concat([R, k], dim=1), dim=1, keepdim=True)

        elif self.config.knowledge_merge == 'concat':
            encoder_input = torch.cat([R, k], dim=-1)

        elif self.config.knowledge_merge == 'mlp':
            if knowledge is not None:
                encoder_input = self.knowledge_merger(torch.cat([R, k], dim=-1))
            else:
                encoder_input = F.relu(R)

        elif self.config.knowledge_merge == 'self-attention':
            R = torch.concat([R, k], dim=1)
            encoder_input = self.attention(R, R, R)
            encoder_input = torch.mean(encoder_input, dim=1, keepdim=True)
        
        elif self.config.knowledge_merge == 'self-attention-2':
            num_R = R.shape[1]
            R = torch.concat([R, k], dim=1)
            encoder_input = self.attention(R, R, R)
            encoder_input = torch.mean(encoder_input[:, :num_R, :], dim=1, keepdim=True) + torch.mean(encoder_input[:, num_R:, :], dim=1, keepdim=True)

        
        q_z_stats = self.encoder(encoder_input)
        
        return q_z_stats, k_deter


class ModulatedLatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.hidden_dim
        if config.latent_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(
                input_dim, 2 * config.hidden_dim
            )

        self.knowledge_encoder = ModulatingKnowledgeEncoder(config)
        
        self.config=config

    def forward(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """

        if knowledge is None:
            modulating_params = None
        else:
            modulating_params = self.knowledge_encoder(knowledge, R)

        if modulating_params is None:
            q_z_stats = self.encoder(R)
        else:
            weights = []
            biases = []
            for name, param in self.encoder.named_parameters():
                if 'weight' in name:
                    weights.append(param)
                elif 'bias' in name:
                    biases.append(param)
            out = R
            for w, b, m in zip(weights[:-1], biases[:-1], modulating_params):
                beta, gamma = m.split(self.config.hidden_dim, dim=-1)
                #beta, gamma = beta.unsqueeze(1), gamma.unsqueeze(1)
                out = F.linear(out, w, b)
                out = out * gamma + beta
                out = self.encoder.activation(out)
            w, b = weights[-1], biases[-1]
            out = F.linear(out, w, b)
            q_z_stats = out

        return q_z_stats, None


class ModulatingKnowledgeEncoder(nn.Module):
    def __init__(self, config):  
        super().__init__()
        if config.text_encoder == 'roberta':
            self.text_encoder = RoBERTa(config, return_cls=False)
        elif config.text_encoder == 'none':
            self.text_encoder = NoEmbedding(config)
        self.knowledge_extractor = MLP(
            input_size=self.text_encoder.dim_model + config.hidden_dim,
            hidden_size=config.hidden_dim * config.latent_encoder_num_hidden,
            num_hidden = config.knowledge_extractor_num_hidden,
            output_size = 2 * config.hidden_dim * config.latent_encoder_num_hidden,
        )
        self.config = config

    def forward(self, knowledge, R):
        k = self.text_encoder(knowledge)
        if self.config.text_encoder == 'roberta':
            k = k.mean(dim=1, keepdim=True)
        elif self.config.text_encoder == 'none':
            k = k.unsqueeze(1)

        modulating_params = self.knowledge_extractor(torch.concat([k, R], dim=-1))
        modulating_params = torch.split(modulating_params, 2 * self.config.hidden_dim, dim=-1)
        return modulating_params


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.decoder_activation == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.GELU()
        if config.path == 'both':
            input_dim = config.hidden_dim * 2 + config.x_transf_dim
        else:
            input_dim = config.hidden_dim + config.x_transf_dim
        self.mlp = MLP(
            input_size=input_dim,
            hidden_size=config.decoder_hidden_dim,
            num_hidden=config.decoder_num_hidden,
            output_size=2 * config.output_dim,
            activation=activation,
        )

    def forward(self, x_target, R_target):
        """
        Decode the target set given the target dependent representation

        R_target [num_samples, bs, num_targets, hidden_dim]
        x_target [bs, num_targets, input_dim]
        """
        x_target = x_target.unsqueeze(0).expand(R_target.shape[0], -1, -1, -1)
        XR_target = torch.cat([x_target, R_target], dim=-1)
        p_y_stats = self.mlp(XR_target)
        return p_y_stats

