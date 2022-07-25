# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 01:36
# @Author      : ssxy00
# @File        : recognition_module.py
# @Description : recognition network

import torch
import torch.nn as nn


class RecognitionModule(nn.Module):
    """(p, c, x) -> z"""

    def __init__(self, embed_dim, z_dim):
        super(RecognitionModule, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(embed_dim * 3, z_dim * 2)

    def forward(self, persona_embedding, context_embedding, response_embedding):
        hidden_states = torch.cat([persona_embedding, context_embedding, response_embedding], dim=1)
        hidden_states = self.linear(hidden_states)
        mu, logvar = hidden_states.split(self.z_dim, dim=1)
        return mu, logvar

class NoEmoRecognitionModule(nn.Module):
    """(p, c, x) -> z"""

    def __init__(self, embed_dim, z_dim):
        super(NoEmoRecognitionModule, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(embed_dim * 2, z_dim * 2)

    def forward(self,context_embedding, response_embedding):
        hidden_states = torch.cat([context_embedding, response_embedding], dim=1)
        hidden_states = self.linear(hidden_states)
        mu, logvar = hidden_states.split(self.z_dim, dim=1)
        return mu, logvar

class EmoRecognitionModule(nn.Module):
    """(p, c) -> z"""
    def __init__(self, embed_dim, z_dim):
        super(EmoRecognitionModule, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(embed_dim * 2, z_dim * 2)

    def forward(self, emotion_embedding, context_embedding):
        hidden_states = torch.cat([emotion_embedding, context_embedding], dim=1)
        hidden_states = self.linear(hidden_states)
        mu, logvar = hidden_states.split(self.z_dim, dim=1)
        return mu, logvar

class ResRecognitionModule(nn.Module):
    """(p, c) -> z"""
    def __init__(self, embed_dim, z_dim):
        super(ResRecognitionModule, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(embed_dim * 2, z_dim * 2)

    def forward(self, context_embedding, response_embedding):
        hidden_states = torch.cat([context_embedding, response_embedding], dim=1)
        hidden_states = self.linear(hidden_states)
        mu, logvar = hidden_states.split(self.z_dim, dim=1)
        return mu, logvar