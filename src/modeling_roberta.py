# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import math
import os
import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class RobertaForMultilabelClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_mode = config.model_mode

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if "cls" in self.model_mode:
            pooled_output = outputs[1].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                hidden_output = outputs[0].view(batch_size, num_chunks*chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size*num_chunks, chunk_size, -1)
            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            # att_weights.masked_fill_((attention_mask.view(batch_size, -1, 1)==0), -math.inf)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
