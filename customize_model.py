from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention


def remove_positional_embeddings(model: AutoModel) -> AutoModel:
    for _, module in model.named_modules():
        if not isinstance(module, BertSdpaSelfAttention):
            continue

        module.position_embedding_type = None

    return model
