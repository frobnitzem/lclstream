import torch
import torch.nn            as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel

# ----------------------------------------------------------------------- #
#  Helper
# ----------------------------------------------------------------------- #
def update_num_channels(model, new_channels=1):
    for child_name, child in model.named_children():
        if hasattr(child, 'num_channels') and child.num_channels == 3:
            print(f"Updating {child_name} num_channels from {child.num_channels} to {new_channels}")
            child.num_channels = new_channels

        # Recursively update submodules
        update_num_channels(child, new_channels)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------- #
#  Model
# ----------------------------------------------------------------------- #
pretrained_model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-huge')


# ----------------------------------------------------------------------- #
#  Adapt
# ----------------------------------------------------------------------- #
update_num_channels(pretrained_model)
pretrained_model.config.num_channels = 1

avg_weight_patch_embd = pretrained_model.vit.embeddings.patch_embeddings.projection.weight.data.mean(dim = 1, keepdim = True)
pretrained_model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(1, 1280, kernel_size=(14, 14), stride=(14, 14))
pretrained_model.vit.embeddings.patch_embeddings.projection.weight.data = avg_weight_patch_embd

avg_weight_decoder_pred = pretrained_model.decoder.decoder_pred.weight.data.view(3, 14, 14, -1).mean(dim = 0).view(14 * 14, -1)
pretrained_model.decoder.decoder_pred = nn.Linear(in_features=512, out_features=588//3, bias=True)
pretrained_model.decoder.decoder_pred.weight.data = avg_weight_decoder_pred

pretrained_model.to(device)


# ----------------------------------------------------------------------- #
#  Example
# ----------------------------------------------------------------------- #
batch_input = {"pixel_values" : torch.rand((1, 1, 224, 224)).to(device)}
batch_outupts = pretrained_model(**batch_input)
