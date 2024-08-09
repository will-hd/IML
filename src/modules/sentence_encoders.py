from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

import torch.nn.functional as F

from typing import Literal
import logging
logger = logging.getLogger(__name__)

from .mlp import MLP

# 'sentence-transformers/all-mpnet-base-v2'
# 'sentence-transformers/all-MiniLM-L6-v2'
# 'sentence-transformers/all-roberta-large-v1' 1024 'pooling_mode_mean_tokens': True (I think)   https://huggingface.co/sentence-transformers/all-roberta-large-v1
# 'sentence-transformers/LaBSE' 768, 'pooling_mode_cls_token': True   https://huggingface.co/sentence-transformers/LaBSE
# 'intfloat/multilingual-e5-large' 1024    https://huggingface.co/intfloat/multilingual-e5-large


class SentenceEncoder(nn.Module):

    def __init__(self,
                 device: Literal['cuda', 'cpu'],
                 pooling_mode: Literal['pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens'],
                ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
        self.model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
        self.dim_model = 768
        self.device = device
        self.pooling_mode = pooling_mode
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self,
               sentences: list[str]
               ) -> torch.Tensor:

        knowledge = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        input_ids = knowledge["input_ids"].to(self.device)
        attention_mask = knowledge["attention_mask"].to(self.device)
        token_type_ids = knowledge["token_type_ids"].to(self.device)

        # print(input_ids.shape)
        # llm_output = self.llm(
        #     input_ids=input_ids.squeeze(1),
        #     attention_mask=attention_mask.squeeze(1),
        #     token_type_ids=token_type_ids.squeeze(1),
        # )
        
        with torch.no_grad():
            model_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,    
            )

        if self.pooling_mode == 'pooling_mode_cls_token':
            output = model_output['pooler_output']
        elif self.pooling_mode == 'pooling_mode_mean_tokens':
            sentence_embeddings = self.mean_pooling(model_output, attention_mask)
            output = F.normalize(sentence_embeddings, p=2, dim=1)
        return output
        

class SentenceKnowledgeEncoder(nn.Module):

    def __init__(self,
                 knowledge_dim: int,
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

        self.text_encoder = SentenceEncoder(device=device)
                        
        self.projection = MLP(
            input_dim=self.text_encoder.dim_model,
            output_dim=knowledge_dim,
            hidden_dim=knowledge_projection_hidden_dim,
            n_h_layers=knowledge_projection_n_h_layers,
            use_bias=use_bias,
            hidden_activation=knowledge_projection_activation
        )
        
    def forward(self,
                desc: tuple[str] | list[str]
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