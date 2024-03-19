import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from decoders import Decoder, TransformerDecoder
from embeddings import Embeddings

def greedy(
        src_mask: Tensor,
        embed: Embeddings,
        decoder: Decoder,
        encoder_output: Tensor,
        trg_input: Tensor,
        model,
        ) -> (np.array, np.array):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    ys = trg_input[:,:1,:].float()

    ys_out = ys
    trg_mask = trg_input != 0.0
    trg_mask = trg_mask.unsqueeze(1)
    max_output_length = trg_input.shape[1]

    if model.just_count_in:
        ys = ys[:,:,-1:]

    for i in range(max_output_length):


        if model.just_count_in:
            ys[:,-1] = trg_input[:, i, -1:]

        else:
            ys[:,-1,-1:] = trg_input[:, i, -1:]

        trg_embed = embed(ys)

        padding_mask = trg_mask[:, :, :i+1, :i+1]

        pad_amount = padding_mask.shape[2] - padding_mask.shape[3]
        padding_mask = (F.pad(input=padding_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)

        with torch.no_grad():
            out, _, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                src_mask=src_mask,
                trg_mask=padding_mask,
            )

            if model.future_prediction != 0:
                out = torch.cat((out[:, :, :out.shape[2] // (model.future_prediction)],out[:,:,-1:]),dim=2)

            if model.just_count_in:
                ys = torch.cat([ys, out[:,-1:,-1:]], dim=1)

            ys = torch.cat([ys, out[:,-1:,:]], dim=1)

            # Add this next predicted frame to the full frame output
            ys_out = torch.cat([ys_out, out[:,-1:,:]], dim=1)

    return ys_out, None

