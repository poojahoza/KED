from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import json
import numpy as np
from utils import last_token_pool, get_detailed_instruct, get_task_string

class GAT(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        # End of trainable weights

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attnt=None):

        out_nodes_features_list = []

        # nodes are the target nodes (entity nodes)
        # neighbors are the source nodes (paragraph nodes and the entity node i.e., self-loop)

        in_nodes_features = nodes

        # apply dropout to all the target nodes as mentioned in the paper
        # shape = (N, FIN), N - number of nodes and FIN - number of input features

        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - number
        # of output features 
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        #print(nodes_features_proj.shape)
        #print(self.scoring_fn_target.shape)

        nodes_features_proj = self.dropout(nodes_features_proj)

        # Attention calculation

        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        #print(scores_target.shape)

        for i in range(len(neighbors)):

            #print(len(neighbors[i]))

            # This is "W" in the paper
            neighbors_features_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_features_proj = self.dropout(neighbors_features_proj)

            # This "a" in the paper
            scores_source = (neighbors_features_proj * self.scoring_fn_source).sum(dim=-1)
            #print(scores_source.shape)

            # get the target node for this neighbor (since the targets could belong to different queries for every neighbor)
            # we then repeat the target node shape to match with the number of neighbors i.e. number of source nodes
            # finally we apply leakyrelu to all
            # Then apply softmax to each edge
            # The above process is to calculate the numerator part of the attention
            #print(scores_target.shape)
            selected_scores_target = torch.index_select(scores_target, 0, torch.tensor([i]).to(self.device)).to(self.device)
            #print(selected_scores_target.shape)
            selected_scores_target = selected_scores_target.repeat_interleave(scores_source.shape[0], dim=0)
            #print(selected_scores_target.shape)
            scores_per_edge = self.leakyReLU(scores_source + selected_scores_target)
            #print(scores_per_edge.shape)
            exp_scores_per_edge = scores_per_edge.exp()
            #print(exp_scores_per_edge.shape)


            # Calculate the denominator.
            # We already have the calculated edge score for the entire neighborhood in exp_scores_per_edge, so just
            # sum it all up

            neighborhood_aware_denominator = exp_scores_per_edge.sum(dim = 0)

            attention_per_edge = exp_scores_per_edge/ (neighborhood_aware_denominator + 1e-16)

            attention_per_edge = attention_per_edge.unsqueeze(-1)

            #print(neighbors_features_proj.shape)
            #print(str(attention_per_edge.shape)+" "+str(neighbors_features_proj.shape))
            #print(attention_per_edge)

            neighbors[i] = neighbors_features_proj*attention_per_edge

            out_nodes = neighbors[i].sum(dim = 0)

            out_nodes_features_list.append(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N,, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


    def init_params(self):

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)



class GATAspects(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, aspects_method='linear', concat=True, activation=nn.ELU(), dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1
        self.aspects_method = aspects_method

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        
        self.aspect_linear_proj = nn.Linear(num_in_features*2, num_out_features)
        
        if self.aspects_method == 'bilinear':
            self.aspect_linear_proj = nn.Bilinear(num_in_features, num_in_features, num_out_features)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        # End of trainable weights

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, aspects, attnt=None):

        out_nodes_features_list = []

        # nodes are the target nodes (entity nodes)
        # neighbors are the source nodes (paragraph nodes and the entity node i.e., self-loop)

        in_nodes_features = nodes

        # apply dropout to all the target nodes as mentioned in the paper
        # shape = (N, FIN), N - number of nodes and FIN - number of input features

        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - number
        # of output features 
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        #print(nodes_features_proj.shape)
        #print(self.scoring_fn_target.shape)

        nodes_features_proj = self.dropout(nodes_features_proj)

        # Attention calculation

        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        #print(scores_target.shape)

        for i in range(len(neighbors)):

            #print(len(neighbors[i]))
            
            # This is "W" for the aspect of the entity
            aspects_features_proj = self.linear_proj(aspects[i]).view(-1, self.num_of_heads, self.num_out_features)
            aspects_features_proj = self.dropout(aspects_features_proj)
            

            # This is "W" in the paper
            neighbors_features_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_features_proj = self.dropout(neighbors_features_proj)
            
            if self.aspects_method == 'linear':
            
                neighbors_aspects_concat_features = torch.cat((neighbors_features_proj, aspects_features_proj), 2)
                neighbors_aspects_features_proj = self.aspect_linear_proj(neighbors_aspects_concat_features)
            
            elif self.aspects_method == 'bilinear':
                
                neighbors_aspects_features_proj = self.aspect_linear_proj(aspects_features_proj, neighbors_features_proj)

            # This "a" in the paper
            scores_source = (neighbors_aspects_features_proj * self.scoring_fn_source).sum(dim=-1)
            #print(scores_source.shape)

            # get the target node for this neighbor (since the targets could belong to different queries for every neighbor)
            # we then repeat the target node shape to match with the number of neighbors i.e. number of source nodes
            # finally we apply leakyrelu to all
            # Then apply softmax to each edge
            # The above process is to calculate the numerator part of the attention
            #print(scores_target.shape)
            selected_scores_target = torch.index_select(scores_target, 0, torch.tensor([i]).to(self.device)).to(self.device)
            #print(selected_scores_target.shape)
            selected_scores_target = selected_scores_target.repeat_interleave(scores_source.shape[0], dim=0)
            #print(selected_scores_target.shape)
            scores_per_edge = self.leakyReLU(scores_source + selected_scores_target)
            #print(scores_per_edge.shape)
            exp_scores_per_edge = scores_per_edge.exp()
            #print(exp_scores_per_edge.shape)


            # Calculate the denominator.
            # We already have the calculated edge score for the entire neighborhood in exp_scores_per_edge, so just
            # sum it all up

            neighborhood_aware_denominator = exp_scores_per_edge.sum(dim = 0)

            attention_per_edge = exp_scores_per_edge/ (neighborhood_aware_denominator + 1e-16)

            attention_per_edge = attention_per_edge.unsqueeze(-1)

            #print(neighbors_features_proj.shape)
            #print(str(attention_per_edge.shape)+" "+str(neighbors_features_proj.shape))
            #print(attention_per_edge)

            neighbors[i] = neighbors_features_proj*attention_per_edge

            out_nodes = neighbors[i].sum(dim = 0)

            out_nodes_features_list.append(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N,, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


    def init_params(self):

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)




class GRNECM(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoated as "w" in the paper) and bias (not mentioned in the
        # paper but present in the official GAT repo)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attention_scores):
        
        in_nodes_features = nodes

        out_nodes_features_list = []


        # shape = (N, FIN) where N-number of nodes and FIN - number of features for each node

        nodes_features_proj = in_nodes_features

        for i in range(len(neighbors)):
            attention_scores[i] = attention_scores[i].unsqueeze(1)
            neighbors[i] = neighbors[i]*attention_scores[i]

            out_nodes = neighbors[i].sum(dim=0)

            out_nodes_features_list.append(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features


class GRN(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoated as "w" in the paper) and bias (not mentioned in the
        # paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False).to(self.device)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attention_scores):
        
        #in_nodes_features = nodes

        out_nodes_features_list = []


        # shape = (N, FIN) where N-number of nodes and FIN - number of features for each node
        # We apply drop out to all of the input nodes as mentioned in the paper

        #in_nodes_features = self.dropout(in_nodes_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features)

        #nodes_features_proj = self.dropout(nodes_features_proj)

        for i in range(len(neighbors)):
            #neighbors_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_proj = self.linear_proj(neighbors[i])
            neighbors_proj = self.dropout(neighbors_proj)
            #print(str(neighbors_proj.shape)+" "+str(attention_scores[i].shape))
            #print(attention_scores[i])
            attention_scores[i] = attention_scores[i].unsqueeze(1)
            #print(attention_scores[i])
            neighbors[i] = neighbors_proj*attention_scores[i]
            #print(attention_scores[i].shape)

            # aggregation of the neighbors i.e. linear combination

            out_nodes = neighbors[i].sum(dim=0)

            out_nodes_features_list.append(out_nodes)

        #out_nodes_features = torch.Tensor(in_nodes_features.shape[0], in_nodes_features.shape[1])
        #print(len(out_nodes_features))
        #print(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)
        #print(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class GRNWeights(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoated as "w" in the paper) and bias (not mentioned in the
        # paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False).to(self.device)
        #self.rating_linear_proj = nn.Linear(1, 1).to(self.device)
        #self.rank_linear_proj = nn.Linear(1, 1).to(self.device)
        self.relevance_linear_proj = nn.Linear(1, 1).to(self.device)
        self.leaky_relu = nn.LeakyReLU(0.2)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attention_scores):
        
        #in_nodes_features = nodes

        out_nodes_features_list = []


        # shape = (N, FIN) where N-number of nodes and FIN - number of features for each node
        # We apply drop out to all of the input nodes as mentioned in the paper

        #in_nodes_features = self.dropout(in_nodes_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features)

        #nodes_features_proj = self.dropout(nodes_features_proj)

        for i in range(len(neighbors)):
            #neighbors_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_proj = self.linear_proj(neighbors[i]) 
            neighbors_proj = self.dropout(neighbors_proj)
            #print(str(neighbors_proj.shape)+" "+str(attention_scores[i].shape))
            #print(attention_scores[i])
            attention_scores[i] = self.leaky_relu(self.relevance_linear_proj(attention_scores[i].unsqueeze(1)))
            
            #attention_scores[i] = self.rating_linear_proj(attention_scores[i][0].unsqueeze(1)) + self.rank_linear_proj(attention_scores[i][1].unsqueeze(1))
            
            #print(attention_scores[i])
            neighbors[i] = neighbors_proj*attention_scores[i]
            #print(attention_scores[i].shape)

            # aggregation of the neighbors i.e. linear combination

            out_nodes = neighbors[i].sum(dim=0)

            out_nodes_features_list.append(out_nodes)

        #out_nodes_features = torch.Tensor(in_nodes_features.shape[0], in_nodes_features.shape[1])
        #print(len(out_nodes_features))
        #print(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)
        #print(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        #nn.init.xavier_uniform_(self.rating_linear_proj.weight)
        #nn.init.xavier_uniform_(self.rank_linear_proj.weight)
        nn.init.xavier_uniform_(self.relevance_linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

class GRNAspects(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, aspects_method='linear', concat=True, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1
        self.aspects_method = aspects_method

        # Trainable weights: linear projection matrix (denoated as "w" in the paper) and bias (not mentioned in the
        # paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        
        self.aspect_linear_proj = nn.Linear(num_in_features*2, num_out_features)
        
        if self.aspects_method == 'bilinear':
            self.aspect_linear_proj = nn.Bilinear(num_in_features, num_in_features, num_out_features)


        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, aspects, attention_scores):
        
        #in_nodes_features = nodes

        out_nodes_features_list = []


        # shape = (N, FIN) where N-number of nodes and FIN - number of features for each node
        # We apply drop out to all of the input nodes as mentioned in the paper

        #in_nodes_features = self.dropout(in_nodes_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.linear_proj(nodes)

        nodes_features_proj = self.dropout(nodes_features_proj)

        for i in range(len(neighbors)):
            
            
            # This is "W" for the aspect of the entity
            aspects_features_proj = self.linear_proj(aspects[i])#.view(-1, self.num_of_heads, self.num_out_features)
            aspects_features_proj = self.dropout(aspects_features_proj)
            

            # This is "W" in the paper
            neighbors_features_proj = self.linear_proj(neighbors[i])#.view(-1, self.num_of_heads, self.num_out_features)
            neighbors_features_proj = self.dropout(neighbors_features_proj)
            
            if self.aspects_method == 'linear':
            
                neighbors_aspects_concat_features = torch.cat((neighbors_features_proj, aspects_features_proj), 1)
                neighbors_aspects_features_proj = self.aspect_linear_proj(neighbors_aspects_concat_features)
            
            elif self.aspects_method == 'bilinear':
                
                neighbors_aspects_features_proj = self.aspect_linear_proj(aspects_features_proj, neighbors_features_proj)

            
            
            #neighbors_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            #neighbors_proj = self.linear_proj(neighbors[i])
            #neighbors_proj = self.dropout(neighbors_proj)
            #print(str(neighbors_proj.shape)+" "+str(attention_scores[i].shape))
            #print(attention_scores[i])
            attention_scores[i] = attention_scores[i].unsqueeze(1)
            #print(attention_scores[i])
            nodes_features_target = torch.index_select(nodes_features_proj, 0, torch.tensor([i]).to(self.device))
            nodes_features_target = nodes_features_target.repeat_interleave(neighbors_aspects_features_proj.shape[0], dim=0)
            features_per_edge = self.leakyReLU(neighbors_aspects_features_proj + nodes_features_target).exp()
            neighbors[i] = features_per_edge*attention_scores[i]
            #print(attention_scores[i].shape)

            # aggregation of the neighbors i.e. linear combination

            out_nodes = neighbors[i].sum(dim=0)

            out_nodes_features_list.append(out_nodes)

        #out_nodes_features = torch.Tensor(in_nodes_features.shape[0], in_nodes_features.shape[1])
        #print(len(out_nodes_features))
        #print(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)
        #print(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.aspect_linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class NeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None

        if experiment == 'grn':
            self.gnn_layer = GRN(50, 50, device)
        else:
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            if len(entity_neighbors_para_text) > 0:

                '''

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()
                '''
                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features), nodes_features
    
    
class GRNECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            layer_flag: int):
        super().__init__()

        self.device = device

        self.gnn_layer = GRNECM(1, 1, device)

        self.rank_score = nn.Linear(1, 1)


    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):


        # constant entity ecm representaiton
        ecm_entity_representation = torch.tensor([0.0]).unsqueeze(1)

        # constant para ecm representation
        ecm_para_representation = torch.tensor([1.0]).unsqueeze(1)

        node_embeddings = ecm_entity_representation.repeat_interleave(entity_emb.shape[0], dim=0).to(self.device)
        #node_embeddings = torch.cat((node_embeddings, repeat_ecm_entity), 1)

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paraembed' in data:
                    entity_neighbors_para_text.append(data['paraembed'])
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            #print(entity_neighbors_para_text)

            if len(entity_neighbors_para_text) > 0:

                # We project down the paragraph representation to 50 dimension

                para_text_embed = torch.from_numpy(np.array(entity_neighbors_para_text)).float().to(self.device)

                #node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))

                if len(entity_neighbors_para_text) == 1:    

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()

                # add ecm paragraph representation to the paragraph BERT representation
                #repeat_ecm_para = ecm_para_representation.repeat_interleave(para_text_embed.shape[0], dim=0).to(self.device)

                para_text_embed = ecm_para_representation.repeat_interleave(para_text_embed.shape[0], dim=0).to(self.device)
                #para_text_embed = torch.cat((para_text_embed, repeat_ecm_para), 1)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)

            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)
        return nodes_features


class TunedBERTModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb).squeeze(1)
        query_embed = self.down_projection(query_emb).squeeze(1)

        rank_scores = torch.matmul(query_embed, ent_embed.transpose(0, 1))

        return rank_scores
    
    
class ParaNeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            feature_selection: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None
        self.para_aggrg = feature_selection

        if experiment == 'paragrn':
            self.gnn_layer = GRN(50, 50, device)
        elif experiment == 'paragat':
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)
        self.feature_selection = nn.Linear(2, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    if self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]*data['parascore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(data['parascore'][0])
                if 'entscore' in data:
                    if self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]*data['entscore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    #entity_neighbors_para_score.append(data['entscore'][0])

            if len(entity_neighbors_para_text) > 0:

                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)

        return self.rank_score(nodes_features)
    
    
class ParaAspectNeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            feature_selection: str,
            aspects_method: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None
        self.para_aggrg = feature_selection
        self.aspects_method = aspects_method

        if experiment == 'paragrnasp':
            self.gnn_layer = GRNAspects(50, 50, device, self.aspects_method)
        elif experiment == 'paragatasp':
            self.gnn_layer = GATAspects(50, 50, device, self.aspects_method)
        self.rank_score = nn.Linear(50, 1)
        self.feature_selection = nn.Linear(2, 1)

    def forward(self, 
                query_emb: torch.Tensor, 
                entity_emb: torch.Tensor, 
                neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        batch_aspects_neighbors_text = []
        
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            neighbors_aspects_embed = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                
                if 'aspembed' in data:
                    asp_embed = torch.from_numpy(np.array(data['aspembed'])).float().to(self.device)
                    asp_embed = self.down_projection(asp_embed).unsqueeze(0)
                    #print(asp_embed.size())
                else:
                    asp_embed = torch.full((1, 50), 0.001).to(self.device)
                neighbors_aspects_embed.append(asp_embed)
                
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    if self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]*data['parascore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(data['parascore'][0])

                    
                if 'entscore' in data:
                    if self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]*data['entscore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    #entity_neighbors_para_score.append(data['entscore'][0])

            if len(entity_neighbors_para_text) > 0:

                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed
                
                
            if len(neighbors_aspects_embed) > 0:
                aspect_embed = torch.cat(neighbors_aspects_embed, dim=0)

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)
            batch_aspects_neighbors_text.append(aspect_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_aspects_neighbors_text, batch_entity_neighbors_score)

        return self.rank_score(nodes_features)
    
    
class ParaTransformerNeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            feature_selection: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None
        self.para_aggrg = feature_selection

        if experiment == 'paragrntrans':
            self.gnn_layer = GRN(50, 50, device)
        elif experiment == 'paragattrans':
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)
        self.feature_selection = nn.Linear(2, 1)
        self.self_representation = nn.Linear(50, 50, bias=False)
        self.self_bias = nn.Parameter(torch.Tensor(50))
        self.self_activation = nn.ELU()
        self.self_dropout = nn.Dropout(p=0.01)

        self.init_params()


    def init_params(self):
        nn.init.xavier_uniform(self.self_representation.weight)
        torch.nn.init.zeros_(self.self_bias)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        batch_entity_self_representation = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))
            
            #self-loop representation in transformer
            batch_entity_self_representation.append(node_embed)


            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    if self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]*data['parascore'][1]).to(self.device))
                    elif self.para_aggrg == 'entrank':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(data['parascore'][0])
                '''
                if 'entscore' in data:
                    if self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]*data['entscore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    #entity_neighbors_para_score.append(data['entscore'][0])
                '''
            
            if len(entity_neighbors_para_text) > 0:

                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)
                score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)
                batch_entity_neighbors_text.append(para_text_embed)
                batch_entity_neighbors_score.append(score_embed)
            else:
                batch_entity_neighbors_text.append(torch.full((1, 50),0.0).to(self.device))
                batch_entity_neighbors_score.append(torch.tensor([0.0]).to(self.device))


        #print(str(len(batch_entity_neighbors_text))+" "+str(len(batch_entity_neighbors_score)))
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)
        '''
        i_81 = torch.index_select(nodes_features, 0, torch.tensor([81]).to(self.device))
        print(i_81)
        '''
        
        #self-loop representations
        self_loop_tensor = torch.cat(batch_entity_self_representation, dim=0)
        self_loop_tensor = self.self_representation(self_loop_tensor)
        self_loop_tensor = self.self_dropout(self_loop_tensor)
        self_loop_tensor = self_loop_tensor*(torch.tensor([1.0]).to(self.device))
        self_loop_tensor += self.self_bias
        self_loop_tensor = self.self_activation(self_loop_tensor)

        '''
        s_81 = torch.index_select(self_loop_tensor, 0, torch.tensor([81]).to(self.device))
        print(s_81)
        '''
        
        assert nodes_features.shape[0] == self_loop_tensor.shape[0]
        assert nodes_features.shape[1] == self_loop_tensor.shape[1]
        
        nodes_features = nodes_features + self_loop_tensor

        '''
        n_81 = torch.index_select(nodes_features, 0, torch.tensor([81]).to(self.device))
        print(n_81)
        print('---------------------------')
        '''

        return self.rank_score(nodes_features)
    
    
class ParaTransformerNeighborsNeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            feature_selection: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None
        self.para_aggrg = feature_selection

        if experiment == 'paragrnneighborstrans':
            self.gnn_layer = GRN(50, 50, device)
        elif experiment == 'paragatneighborstrans':
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)
        self.feature_selection = nn.Linear(2, 1)
        self.self_representation = nn.Linear(50, 50, bias=False)
        self.self_bias = nn.Parameter(torch.Tensor(50))
        self.self_activation = nn.ELU()
        self.self_dropout = nn.Dropout(p=0.01)

        self.init_params()


    def init_params(self):
        nn.init.xavier_uniform(self.self_representation.weight)
        torch.nn.init.zeros_(self.self_bias)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        batch_entity_self_representation = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))
            
            #self-loop representation in transformer
            batch_entity_self_representation.append(node_embed)


            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    if self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['parascore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]*data['parascore'][1]).to(self.device))
                    elif self.para_aggrg == 'entrank':
                        entity_neighbors_para_score.append(torch.tensor(data['parascore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(data['parascore'][0])
                '''
                if 'entscore' in data:
                    if self.para_aggrg == 'max':
                        entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'linear':
                        entity_neighbors_para_score.append(self.feature_selection(torch.tensor(data['entscore']).to(self.device)))
                    elif self.para_aggrg == 'prod':
                        entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]*data['entscore'][1]).to(self.device))
                    #entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore']).to(self.device)))
                    #entity_neighbors_para_score.append(data['entscore'][0])
                '''
            
            if len(entity_neighbors_para_text) > 0:

                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)
                score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)
                batch_entity_neighbors_text.append(para_text_embed)
                batch_entity_neighbors_score.append(score_embed)
            else:
                batch_entity_neighbors_text.append(torch.full((1, 50),0.0).to(self.device))
                batch_entity_neighbors_score.append(torch.tensor([0.0]).to(self.device))


        #print(str(len(batch_entity_neighbors_text))+" "+str(len(batch_entity_neighbors_score)))
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)
        '''
        i_81 = torch.index_select(nodes_features, 0, torch.tensor([81]).to(self.device))
        print(i_81)
        '''
        '''
        #self-loop representations
        self_loop_tensor = torch.cat(batch_entity_self_representation, dim=0)
        self_loop_tensor = self.self_representation(self_loop_tensor)
        self_loop_tensor = self.self_dropout(self_loop_tensor)
        self_loop_tensor = self_loop_tensor*(torch.tensor([1.0]).to(self.device))
        self_loop_tensor += self.self_bias
        self_loop_tensor = self.self_activation(self_loop_tensor)
        
        assert nodes_features.shape[0] == self_loop_tensor.shape[0]
        assert nodes_features.shape[1] == self_loop_tensor.shape[1]
        
        nodes_features = nodes_features + self_loop_tensor
        '''


        return self.rank_score(nodes_features)
    
class ParaTransformerParaEntityModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.rank_score = nn.Linear(50, 1)
        self.self_representation = nn.Linear(50, 50, bias=False)
        self.self_dropout = nn.Dropout(p=0.01)
        self.self_bias = nn.Parameter(torch.Tensor(50))
        self.self_activation = nn.ELU()

        self.init_params()


    def init_params(self):
        nn.init.xavier_uniform(self.self_representation.weight)
        torch.nn.init.zeros_(self.self_bias)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)
        
        #print(query_embed.shape)
        #print(ent_embed.shape)
        
        batch_ent_neighbors_list = []
        
        for i, n in enumerate(neighbors):
            neighbor_entities_tensor = torch.full((1, 50),1.0).to(self.device)
            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed).unsqueeze(0)
                    neighbor_entities_tensor = neighbor_entities_tensor * para_embed
            #print(neighbor_entities_tensor.shape)
            batch_ent_neighbors_list.append(neighbor_entities_tensor)
        #para_embed = self.down_projection(para_embed).unsqueeze(0)
            
        batch_ent_neighbors_tensor = torch.cat(batch_ent_neighbors_list, dim=0)
        node_embeddings = torch.squeeze(query_embed) * ent_embed * batch_ent_neighbors_tensor
        
        self_node_features = self.self_representation(node_embeddings)
        self_node_features = self.self_dropout(self_node_features)
        self_node_features += self.self_bias
        self_node_features = self.self_activation(self_node_features)

        return self.rank_score(self_node_features)
    
    
class ParaTransformerTunedBertModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.rank_score = nn.Linear(50, 1)
        self.self_representation = nn.Linear(50, 50, bias=False)
        self.self_dropout = nn.Dropout(p=0.01)
        self.self_bias = nn.Parameter(torch.Tensor(50))
        self.self_activation = nn.ELU()

        self.init_params()


    def init_params(self):
        nn.init.xavier_uniform(self.self_representation.weight)
        torch.nn.init.zeros_(self.self_bias)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed
        
        self_node_features = self.self_representation(node_embeddings)
        self_node_features = self.self_dropout(self_node_features)
        self_node_features += self.self_bias
        self_node_features = self.self_activation(self_node_features)

        return self.rank_score(self_node_features)
    
class NeuralECMTokenModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str):   
        super().__init__()
        
        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        self.gnn_layer = None
        
        self.model = 'distilbert-base-uncased'
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.config = AutoConfig.from_pretrained(self.model)
        self.bert = AutoModel.from_pretrained(self.model)

        if experiment == 'grnall':
            self.gnn_layer = GRN(50, 50, device)
        elif experiment == 'gatall':
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List, entity_text: List, query_text: List):
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.bert.eval()
        
        tokens = self.tokenizer.batch_encode_plus(entity_text, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        entity_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        entity_emb1 = entity_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
        
        #Expand query_emb to have same shape as entity_emb: query_Emb (1000,768) and entity_Emb = (1000, 512,768)
        query_emb_reshaped = query_emb.unsqueeze(1).expand(-1, entity_emb1.size(1), -1)

        #Normalize entity and query embeddings
        query_emb_norm = F.normalize(query_emb_reshaped, dim=-1)
        entity_emb_norm = F.normalize(entity_emb1, dim=-1)

        cos_sim = torch.matmul(entity_emb_norm, query_emb_norm.transpose(-1, -2))
        #print(cos_sim.shape)
        #print(torch.transpose(cos_sim, 0, 1).shape)
        #print(cos_sim)
        split_cos_sim = torch.split(cos_sim, 1, dim=-1)[0] #split along dimension 2
        #print(split_cos_sim)
        #print(cos_sim.view(32, 511, 1))
        #print(torch.transpose(split_cos_sim, 1, 2))
        #print(entity_text)
        transposed_cos_sim = torch.transpose(split_cos_sim, 1, 2)

        best_token_indices = torch.argmax(transposed_cos_sim, dim=-1) #along dimension 2
        best_token_indices_flatten = torch.flatten(best_token_indices, 0)

        entity_best_token = entity_emb1[torch.arange(entity_emb1.size(0)), best_token_indices_flatten]
        #print('entity shape : '+str(entity_emb1.shape)+' '+str(query_emb.shape)+" "+str(entity_best_token.shape)+" "+str(cos_sim.shape)+" "+str(best_token_indices.shape))

        ent_embed = self.down_projection(entity_best_token)
        query_embed = self.down_projection(query_emb)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))
            i_query_embed_norm = torch.index_select(query_emb_norm.to(self.device), 0, torch.tensor([i]).to(self.device)).to(self.device)

            for data in n:
                if 'paratext' in data:
                    #para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)


                    tokens = self.tokenizer.batch_encode_plus([data['paratext']], return_tensors='pt', truncation=True, padding=True)
                    input_ids = tokens.input_ids.to(self.device)
                    attention_mask = tokens.attention_mask.to(self.device)
                    para_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    para_embed = para_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
                    #print(para_embed.shape)

                    para_embed_norm = F.normalize(para_embed, dim=-1)
                    #print(para_embed_norm.shape)
                    #print(query_emb_norm.shape)
                    #print(i_query_embed_norm.squeeze().shape)
                    para_cos_sim = torch.matmul(para_embed_norm, i_query_embed_norm.squeeze().transpose(0, -1))
                    #print(para_cos_sim.shape)
                    #print(para_cos_sim)
                    split_para_cos_sim = torch.split(para_cos_sim, 1, dim=-1)[0].squeeze()
                    #print(split_para_cos_sim.shape)
                    #transposed_para_cos_sim = torch.transpose(split_para_cos_sim, -1, -2)
                    #print(transposed_para_cos_sim.shape)
                    best_token_indices = torch.argmax(split_para_cos_sim)
                    #print(best_token_indices)
                    #best_token_indices_flatten = torch.flatten(best_token_indices, 0)
                    #print(best_token_indices_flatten.shape)
                    para_best_token = para_embed[best_token_indices]
                    #print(para_best_token.shape)

                    para_embed = self.down_projection(para_best_token).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            if len(entity_neighbors_para_text) > 0:

                '''

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()
                '''
                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features)


class ParaSymbolsTokenModel1(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            batch_size: int):   
        super().__init__()
        
        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50).to(self.device)
        self.gnn_layer = None
        self.max_length = 4096
        
        self.model_name = 'intfloat/e5-mistral-7b-instruct'
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
                return_dict=True
            )
        
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        #self.config = AutoConfig.from_pretrained(self.model)
        #self.bert = AutoModel.from_pretrained(self.model)

        # if experiment == 'grnall':
        #     self.gnn_layer = GRN(50, 50, device)
        # elif experiment == 'gatall':
        #     self.gnn_layer = GAT(50, 50, device)
        self.gnn_layer = GRN(50, 50, device)
        self.rank_score = nn.Linear(50, 1).to(self.device)
        
    def get_model_embeddings(self, input_texts: List):
        batch_dict = self.tokenizer(input_texts, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to(self.device)
        
        outputs = self.model(**batch_dict)
        
        embeddings = last_token_pool(outputs.hidden_states[1], batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
        

    def forward(self, neighbors: List, entity_text: List, query_text: List, task: str):
        
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        # self.bert.eval()
        
        self.model.eval()
        
        task_string = get_task_string(task)
        queries = [get_detailed_instruct(task_string, q) for q in query_text]
        
        input_texts = queries + entity_text
        
        embeddings = self.get_model_embeddings(input_texts)
        
        query_embed = embeddings[:self.batch_size]
        entity_embed = embeddings[self.batch_size:]
        
        '''
        tokens = self.tokenizer.batch_encode_plus(entity_text, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        entity_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        entity_emb1 = entity_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
        
        #Expand query_emb to have same shape as entity_emb: query_Emb (1000,768) and entity_Emb = (1000, 512,768)
        query_emb_reshaped = query_emb.unsqueeze(1).expand(-1, entity_emb1.size(1), -1)

        #Normalize entity and query embeddings
        query_emb_norm = F.normalize(query_emb_reshaped, dim=-1)
        entity_emb_norm = F.normalize(entity_emb1, dim=-1)

        cos_sim = torch.matmul(entity_emb_norm, query_emb_norm.transpose(-1, -2))
        #print(cos_sim.shape)
        #print(torch.transpose(cos_sim, 0, 1).shape)
        #print(cos_sim)
        split_cos_sim = torch.split(cos_sim, 1, dim=-1)[0] #split along dimension 2
        #print(split_cos_sim)
        #print(cos_sim.view(32, 511, 1))
        #print(torch.transpose(split_cos_sim, 1, 2))
        #print(entity_text)
        transposed_cos_sim = torch.transpose(split_cos_sim, 1, 2)

        best_token_indices = torch.argmax(transposed_cos_sim, dim=-1) #along dimension 2
        best_token_indices_flatten = torch.flatten(best_token_indices, 0)

        entity_best_token = entity_emb1[torch.arange(entity_emb1.size(0)), best_token_indices_flatten]
        #print('entity shape : '+str(entity_emb1.shape)+' '+str(query_emb.shape)+" "+str(entity_best_token.shape)+" "+str(cos_sim.shape)+" "+str(best_token_indices.shape))
        
        
        ent_embed = self.down_projection(entity_best_token)
        query_embed = self.down_projection(query_emb)
        '''
        
        entity_embed = self.down_projection(entity_embed)
        query_embed = self.down_projection(query_embed)

        node_embeddings = torch.squeeze(query_embed) * entity_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            '''
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))
            i_query_embed_norm = torch.index_select(query_emb_norm.to(self.device), 0, torch.tensor([i]).to(self.device)).to(self.device)
            '''
            target_node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paratext' in data:
                    #para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)


                    '''
                    tokens = self.tokenizer.batch_encode_plus([data['paratext']], return_tensors='pt', truncation=True, padding=True)
                    input_ids = tokens.input_ids.to(self.device)
                    attention_mask = tokens.attention_mask.to(self.device)
                    para_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    para_embed = para_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
                    #print(para_embed.shape)

                    para_embed_norm = F.normalize(para_embed, dim=-1)
                    #print(para_embed_norm.shape)
                    #print(query_emb_norm.shape)
                    #print(i_query_embed_norm.squeeze().shape)
                    para_cos_sim = torch.matmul(para_embed_norm, i_query_embed_norm.squeeze().transpose(0, -1))
                    #print(para_cos_sim.shape)
                    #print(para_cos_sim)
                    split_para_cos_sim = torch.split(para_cos_sim, 1, dim=-1)[0].squeeze()
                    #print(split_para_cos_sim.shape)
                    #transposed_para_cos_sim = torch.transpose(split_para_cos_sim, -1, -2)
                    #print(transposed_para_cos_sim.shape)
                    best_token_indices = torch.argmax(split_para_cos_sim)
                    #print(best_token_indices)
                    #best_token_indices_flatten = torch.flatten(best_token_indices, 0)
                    #print(best_token_indices_flatten.shape)
                    para_best_token = para_embed[best_token_indices]
                    #print(para_best_token.shape)
                    '''
                    
                    embeddings = self.get_model_embeddings([data['paratext']])
                    para_embed = self.down_projection(embeddings).unsqueeze(0)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            if len(entity_neighbors_para_text) > 0:

                '''

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()
                '''
                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                '''
                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
                '''
                para_text_embed = torch.cat((para_text_embed, target_node_embed), 0)
            else:
                para_text_embed = target_node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features)
    
    
class ParaSymbolsTokenModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str,
            batch_size: int):   
        super().__init__()
        
        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50).to(self.device)
        self.gnn_layer = None
        self.max_length = 4096
        self.batch_size = batch_size
        
        #self.model_name = 'intfloat/e5-mistral-7b-instruct'
        self.model_name = 'google/flan-t5-large'
        
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #         self.model_name,
        #         load_in_4bit=True,
        #         quantization_config=bnb_config,
        #         torch_dtype=torch.bfloat16,
        #         device_map="auto",
        #         trust_remote_code=True,
        #         output_hidden_states=True,
        #         return_dict=True
        #     )
        
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        #self.config = AutoConfig.from_pretrained(self.model)
        #self.bert = AutoModel.from_pretrained(self.model)

        # if experiment == 'grnall':
        #     self.gnn_layer = GRN(50, 50, device)
        # elif experiment == 'gatall':
        #     self.gnn_layer = GAT(50, 50, device)
        self.gnn_layer = GRN(50, 50, device)
        self.rank_score = nn.Linear(50, 1).to(self.device)
        
    # def get_model_embeddings(self, input_texts: List):
    #     batch_dict = self.tokenizer(input_texts, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    #     batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    #     batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to(self.device)
        
    #     outputs = self.model(**batch_dict)
        
    #     embeddings = last_token_pool(outputs.hidden_states[1], batch_dict['attention_mask'])
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
        
    #     return embeddings
    
    def get_model_embeddings(self, input_texts: List):
        
        batch_dict = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        output = self.model.encoder(input_ids=batch_dict['input_ids'],
                                    attention_mask=batch_dict['attention_mask'],
                                    return_dict=True)
        #pooled_sentence = (output.last_hidden_state * batch_dict['attention_mask'].unsqueeze(-1)).sum(dim=-2) / batch_dict['attention_mask'].sum(dim=-1)
        pooled_sentence = output.last_hidden_state
        
        pooled_sentence = torch.mean(pooled_sentence, dim=1)
        
        return pooled_sentence
        

    def forward(self, neighbors: List, entity_text: List, query_text: List, task: str):
        
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        # self.bert.eval()
        
        task_string = get_task_string(task)
        queries = [get_detailed_instruct(task_string, q) for q in query_text]
        queries_length = len(queries)
        
        input_texts = queries + entity_text
        
        embeddings = self.get_model_embeddings(input_texts)
        
        query_embed = embeddings[:queries_length]
        entity_embed = embeddings[queries_length:]
        
        '''
        tokens = self.tokenizer.batch_encode_plus(entity_text, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        entity_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        entity_emb1 = entity_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
        
        #Expand query_emb to have same shape as entity_emb: query_Emb (1000,768) and entity_Emb = (1000, 512,768)
        query_emb_reshaped = query_emb.unsqueeze(1).expand(-1, entity_emb1.size(1), -1)

        #Normalize entity and query embeddings
        query_emb_norm = F.normalize(query_emb_reshaped, dim=-1)
        entity_emb_norm = F.normalize(entity_emb1, dim=-1)

        cos_sim = torch.matmul(entity_emb_norm, query_emb_norm.transpose(-1, -2))
        #print(cos_sim.shape)
        #print(torch.transpose(cos_sim, 0, 1).shape)
        #print(cos_sim)
        split_cos_sim = torch.split(cos_sim, 1, dim=-1)[0] #split along dimension 2
        #print(split_cos_sim)
        #print(cos_sim.view(32, 511, 1))
        #print(torch.transpose(split_cos_sim, 1, 2))
        #print(entity_text)
        transposed_cos_sim = torch.transpose(split_cos_sim, 1, 2)

        best_token_indices = torch.argmax(transposed_cos_sim, dim=-1) #along dimension 2
        best_token_indices_flatten = torch.flatten(best_token_indices, 0)

        entity_best_token = entity_emb1[torch.arange(entity_emb1.size(0)), best_token_indices_flatten]
        #print('entity shape : '+str(entity_emb1.shape)+' '+str(query_emb.shape)+" "+str(entity_best_token.shape)+" "+str(cos_sim.shape)+" "+str(best_token_indices.shape))
        
        
        ent_embed = self.down_projection(entity_best_token)
        query_embed = self.down_projection(query_emb)
        '''
        
        entity_embed = self.down_projection(entity_embed)
        query_embed = self.down_projection(query_embed)

        node_embeddings = query_embed * entity_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            '''
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))
            i_query_embed_norm = torch.index_select(query_emb_norm.to(self.device), 0, torch.tensor([i]).to(self.device)).to(self.device)
            '''
            target_node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paratext' in data:
                    #para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)


                    '''
                    tokens = self.tokenizer.batch_encode_plus([data['paratext']], return_tensors='pt', truncation=True, padding=True)
                    input_ids = tokens.input_ids.to(self.device)
                    attention_mask = tokens.attention_mask.to(self.device)
                    para_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    para_embed = para_outputs.last_hidden_state[:, 1:, :].squeeze().to(self.device)
                    #print(para_embed.shape)

                    para_embed_norm = F.normalize(para_embed, dim=-1)
                    #print(para_embed_norm.shape)
                    #print(query_emb_norm.shape)
                    #print(i_query_embed_norm.squeeze().shape)
                    para_cos_sim = torch.matmul(para_embed_norm, i_query_embed_norm.squeeze().transpose(0, -1))
                    #print(para_cos_sim.shape)
                    #print(para_cos_sim)
                    split_para_cos_sim = torch.split(para_cos_sim, 1, dim=-1)[0].squeeze()
                    #print(split_para_cos_sim.shape)
                    #transposed_para_cos_sim = torch.transpose(split_para_cos_sim, -1, -2)
                    #print(transposed_para_cos_sim.shape)
                    best_token_indices = torch.argmax(split_para_cos_sim)
                    #print(best_token_indices)
                    #best_token_indices_flatten = torch.flatten(best_token_indices, 0)
                    #print(best_token_indices_flatten.shape)
                    para_best_token = para_embed[best_token_indices]
                    #print(para_best_token.shape)
                    '''
                    
                    embeddings = self.get_model_embeddings([data['paratext']])
                    para_embed = self.down_projection(embeddings)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            if len(entity_neighbors_para_text) > 0:

                '''

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()
                '''
                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                '''
                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
                '''
                para_text_embed = torch.cat((para_text_embed, target_node_embed), 0)
            else:
                para_text_embed = target_node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features)
    
class ParaSymbolsEmbeddingModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)
        #self.gnn_layer = GRN(50, 50, device)
        #self.gnn_layer = GAT(50, 50, device)
        self.gnn_layer = GRNWeights(50, 50, device)

        self.rank_score = nn.Linear(50, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb).squeeze(1) #shape (batch_size, 50)
        query_embed = self.down_projection(query_emb).squeeze(1) # shape (batch_size, 50)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))
            i_query_embed = torch.index_select(query_embed, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paraembed' in data:
                    para_embed = torch.from_numpy(np.array(data['paraembed'])).float().to(self.device)
                    para_embed = self.down_projection(para_embed)

                    para_embeddings = i_query_embed*para_embed

                    entity_neighbors_para_text.append(para_embeddings)
                    
                    #entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]).to(self.device))
                    entity_neighbors_para_score.append(torch.max(torch.tensor(data['parascore'][1:])).to(self.device))
                    #entity_neighbors_para_score.append(torch.Tensor([data['parascore'][0],max(data['parascore'][1:])]).unsqueeze(0).to(self.device))
                    
                if 'entscore' in data:
                    #entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]).to(self.device))
                    entity_neighbors_para_score.append(torch.max(torch.tensor(data['entscore'][1:])).to(self.device))
                    #entity_neighbors_para_score.append(torch.Tensor([data['entscore'][0], max(data['entscore'][1:])]).unsqueeze(0).to(self.device))
                    
            if len(entity_neighbors_para_text) > 0:

                
                entity_neighbors_para_text.append(node_embed)
                para_text_embed = torch.cat(entity_neighbors_para_text, dim=0)

                #para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)
            #score_embed = torch.cat(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)

        return self.rank_score(nodes_features)


class ParaSymbolsSelfRatingWithoutGNNEmbeddingModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            experiment: str):
        super().__init__()

        self.device = device
        self.down_projection = nn.Linear(query_input_emb_dim, 50)

        self.rank_score = nn.Linear(50, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.down_projection(entity_emb).squeeze(1) #shape (batch_size, 50)
        query_embed = self.down_projection(query_emb).squeeze(1) # shape (batch_size, 50)

        node_embeddings = torch.squeeze(query_embed) * ent_embed

        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_score = []
            
            for data in n:
                if 'paraembed' in data:
                    entity_neighbors_para_score.append(torch.tensor(data['parascore'][0]).to(self.device))
                if 'entscore' in data:
                    entity_neighbors_para_score.append(torch.tensor(data['entscore'][0]).to(self.device))

            batch_entity_neighbors_score.append(sum(entity_neighbors_para_score))
            
        batch_neighbors_tensor = torch.Tensor(batch_entity_neighbors_score).to(self.device)
        
        nodes_features = node_embeddings*batch_neighbors_tensor


        return self.rank_score(nodes_features)
