import itertools
import torch


class MarianDecoderWrapped(torch.nn.Module):
    def __init__(self, decoder):
        super(MarianDecoderWrapped, self).__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        return self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False
        )


class MarianDecoderPkvWrapped(torch.nn.Module):
    def __init__(self, decoder):
        super(MarianDecoderPkvWrapped, self).__init__()
        self.decoder = decoder

    def group(self, lst):
        return tuple(zip(*[itertools.islice(lst, i, None, 4) for i in range(4)]))

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        *past_key_values
    ):
        past_key_values = self.group(past_key_values)

        return self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False
        )
