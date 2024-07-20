import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from typing import Literal
import logging
logger = logging.getLogger(__name__)

from .mlp import MLP

        
class RoBERTa(nn.Module):
    def __init__(self,
                 return_cls: bool,
                 device: Literal['cuda', 'cpu'],
                 tune_llm_layer_norms: bool,
                 freeze_llm: bool,
                 ):
        super(RoBERTa, self).__init__()

        self.dim_model = 768
        self.llm = RobertaModel.from_pretrained("roberta-base")

        self.return_cls = return_cls
        self.device =device

        if freeze_llm:
            logger.debug("Freezing LLM parameters")
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        if tune_llm_layer_norms:
            logger.debug("Allowing LLM LayerNorm parameters to be trained")
            for name, param in self.llm.named_parameters():
                if "LayerNorm" in name:
                    param.requires_grad = True


        logging.debug("Training pooler layer parameters")
        for name, param in self.llm.named_parameters():
            if name == "pooler.dense.weight" or name == "pooler.dense.bias":
                param.requires_grad = True


        self.tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base', truncation=True, do_lower_case=True
        )

        if return_cls:
            logging.info("LLM output is CLS token")
        else:
            logging.info("LLM output is last hidden state")

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


class RoBERTaKnowledgeEncoder(nn.Module):

    def __init__(self,
                 knowledge_dim: int,
                 return_cls: bool,
                 freeze_llm: bool,
                 tune_llm_layer_norms: bool,
                 use_bias: bool,
                 device: Literal['cuda', 'cpu'],
                 knowledge_projection_n_h_layers: int,
                 knowledge_projection_hidden_dim: int,
                 knowledge_projection_activation: nn.Module,
                 ) -> None:
        """
        Parameters
        ----------
        knowledge_dim : int
            Dimension of the knowledge representation
        return_cls : bool
            Use CLS token as sentence pooling
        freeze_llm : bool
            Freeze LLM parameters
        tune_llm_layer_norms : bool
            Set LayerNorm parameters to be trainable
        knowledge_projection_n_h_layers : int
            Number of hidden layers in the projection MLP 
        """
        super().__init__()

        self.text_encoder = RoBERTa(
            return_cls=return_cls,
            device=device,
            tune_llm_layer_norms=tune_llm_layer_norms,
            freeze_llm=freeze_llm
        )
                        
        self.projection = MLP(
            input_dim=self.text_encoder.dim_model,
            output_dim=knowledge_dim,
            hidden_dim=knowledge_projection_hidden_dim,
            n_h_layers=knowledge_projection_n_h_layers,
            use_bias=use_bias,
            hidden_activation=knowledge_projection_activation
        )

        
    def forward(self,
                desc: tuple[str]
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        desc : tuple[str]
            Textual knowledge as a tuple of strings 
        
        Returns
        -------
        k : torch.Tensor
            Shape (batch_size, 1, knowledge_dim) 
            The aggregated knowledge representation
        """
        text_repr = self.text_encoder(desc)
        k = self.projection(text_repr) # Shape (batch_size, knowledge_dim)
        k = k.unsqueeze(1)
        return k # Shape (batch_size, 1, knowledge_dim)


class KnowledgeEncoder(nn.Module):

    def __init__(self,
                 knowledge_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 n_h_layers_phi: int,
                 n_h_layers_rho: int,
                 only_use_linear: bool,
                 use_bias: bool
                 ) -> None:
        super().__init__()

        self._only_use_linear = only_use_linear

        if only_use_linear:
            self.linear = nn.Linear(knowledge_dim, hidden_dim, bias=use_bias)
        
        else: 
            self.phi = MLP(input_dim=knowledge_dim,
                                output_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_phi,
                                use_bias=use_bias,
                                hidden_activation=nn.GELU())
            
            self.rho = MLP(input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                n_h_layers=n_h_layers_rho,
                                use_bias=use_bias,
                                hidden_activation=nn.GELU())
        
    def forward(self,
                knowledge: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        knowledge : torch.Tensor
            Shape (batch_size, num_knowledge_points, knowledge_dim)
        
        Returns
        -------
        k : torch.Tensor
            Shape (batch_size, 1, hidden_dim) 
            The aggregated knowledge representation
        """
        if self._only_use_linear:
            k = self.linear(knowledge) # Shape (batch_size, 1, hidden_dim)
            assert k.size(1) == 1
        
        else:
            knowledge_encoded = self.phi(knowledge) # Shape (batch_size, num_knowledge_points, hidden_dim)

            mean_repr = torch.mean(knowledge_encoded, dim=1, keepdim=True) # Shape (batch_size, 1, hidden_dim)

            k = self.rho(mean_repr) # Shape (batch_size, 1, hidden_dim)

        return k # Shape (batch_size, 1, hidden_dim)
