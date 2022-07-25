# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 02:43
# @Author      : ssxy00
# @File        : cvae_decoder.py
# @Description :

import torch.nn as nn
import torch


class CVAEDecoder(nn.Module):
    """decoder, reconstruct response"""

    def __init__(self, core_module, pad_id, z_dim, embed_dim, inject_type):
        super(CVAEDecoder, self).__init__()
        self.inject_type = inject_type
        self.core_module = core_module
        self.pad_id = pad_id
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        # initialize and tie lm head
        self.lm_head = nn.Linear(embed_dim, core_module.wte.weight.size(0), bias=False)
        self.lm_head.weight = core_module.wte.weight

        # initialize latent head
        if inject_type == "memory":
            self.latent_head = nn.Linear(z_dim, embed_dim * core_module.config.n_layer, bias=False)
        if inject_type == "embedding":
            self.latent_head = nn.Linear(z_dim, embed_dim, bias=False)

    def forward(self, input_ids, type_ids, latent_sample=None):
        if latent_sample is None:
            # decoder
            hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids)[0]
        else:
            if self.inject_type == "memory":
                # cvae_memory
                extra_hidden_states = self.latent_head(latent_sample)
                extra_hidden_states = [h.unsqueeze(1) for h in extra_hidden_states.split(self.embed_dim, dim=1)]
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,
                                                 extra_hidden_states=extra_hidden_states)[0]
            elif self.inject_type == "embedding":
                # cvae embedding
                extra_embedding = self.latent_head(latent_sample).unsqueeze(1)
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,
                                                 extra_embedding=extra_embedding)[0]
            else:
                raise ValueError("unknown injection type")

        logits = self.lm_head(hidden_states)
        return logits

class UACVAEConvDecoder(nn.Module):
    """decoder, reconstruct response"""

    def __init__(self, core_module, pad_id, z_dim, embed_dim, inject_type):
        super(UACVAEConvDecoder, self).__init__()
        self.inject_type = inject_type
        self.core_module = core_module
        self.pad_id = pad_id
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.inter_dim = z_dim * 2

        # initialize and tie lm head
        self.lm_head = nn.Linear(embed_dim, core_module.wte.weight.size(0), bias=False)
        self.lm_head.weight = core_module.wte.weight

        if inject_type == "embedding":
            #conv4
            self.conv = nn.Conv1d(2, 3, kernel_size=1)
            self.latent_head_var = nn.Linear(z_dim, embed_dim,bias=False)  # self.latent_head_var = nn.Linear(z_dim, embed_dim, bias=False)
            self.latent_head_LS = nn.Linear(z_dim, embed_dim, bias=False)
            # self.latent_head_Comb1 = nn.Linear(embed_dim, embed_dim, bias=False) #conv3
            self.latent_head_Comb1 = nn.Linear(embed_dim*3, embed_dim, bias=False)
            #self.latent_head_Comb2 = nn.Linear(embed_dim*2, embed_dim, bias=False)

    def forward(self, input_ids, type_ids, latent_sample=None,  logvar = None):
        if latent_sample is None:
            # decoder
            hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids)[0]
        else:
            if self.inject_type == "embedding":

                extra_embedding_LS = self.latent_head_var(latent_sample)
                extra_embedding_var = self.latent_head_LS(logvar)
                stacked = torch.stack([extra_embedding_LS, extra_embedding_var], dim=1)
                conv_embed = self.conv(stacked)
                extra_embedding = self.latent_head_Comb1(torch.flatten(conv_embed, start_dim=1)).unsqueeze(1)
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,extra_embedding=extra_embedding)[0]
            else:
                raise ValueError("unknown injection type")

        logits = self.lm_head(hidden_states)
        return logits

class UACVAEMlpDecoder(nn.Module):
    """decoder, reconstruct response"""

    def __init__(self, core_module, pad_id, z_dim, embed_dim, inject_type):
        super(UACVAEMlpDecoder, self).__init__()
        self.inject_type = inject_type
        self.core_module = core_module
        self.pad_id = pad_id
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.inter_dim = z_dim * 2
        # initialize and tie lm head
        self.lm_head = nn.Linear(embed_dim, core_module.wte.weight.size(0), bias=False)
        self.lm_head.weight = core_module.wte.weight
        if inject_type == "embedding":
            self.latent_head_var = nn.Linear(z_dim, embed_dim, bias=False)#self.latent_head_var = nn.Linear(z_dim, embed_dim, bias=False)
            self.latent_head_LS = nn.Linear(z_dim, embed_dim, bias=False)
            #self.relu = nn.ReLU()
            #self.batch_norm = nn.BatchNorm1d(self.inter_dim)
            #self.latent_head_Comb =  nn.Linear(self.inter_dim, embed_dim, bias=False)

    def forward(self, input_ids, type_ids, latent_sample=None,  logvar = None):
        if latent_sample is None:
            # decoder
            hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids)[0]
        else:
            if self.inject_type == "embedding":
                # cvae embedding
                #extra_embedding_LS = self.latent_head_var(latent_sample).unsqueeze(1)
                #extra_embedding_var = self.latent_head_LS(logvar).unsqueeze(1)
                extra_embedding_LS = self.latent_head_var(latent_sample)
                extra_embedding_var = self.latent_head_LS(logvar)
                extra_embedding = (extra_embedding_LS + extra_embedding_var).unsqueeze(1)
                #extra_embedding = self.decConv(extra_embedding_LS,extra_embedding_var).unsqueeze(1)
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,extra_embedding=extra_embedding)[0]
            else:
                raise ValueError("unknown injection type")

        logits = self.lm_head(hidden_states)
        return logits
