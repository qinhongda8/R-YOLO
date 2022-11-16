import torch
import torch.nn.functional as F

def prepare_masks(targets, is_source_mark):
        masks = []
        # targets = torch.tensor([1,2,3,4,5,6,7,8]).to("cuda:0")
        for targets_per_image in targets:
            is_source = targets_per_image
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source_mark else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

def loss_eval( domain_1, target, is_source_mark):
        masks = prepare_masks(target, is_source_mark)
        masks = torch.cat(masks, dim=0)
        N, A, H, W = domain_1.shape
        da_img_per_level = domain_1.permute(0, 2, 3, 1)
        da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
        masks = masks.bool()
        # print("masks: ",masks)
        da_img_label_per_level[masks, :] = 1

        da_img_per_level = da_img_per_level.reshape(N, -1)
        da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        da_img_labels_flattened = da_img_label_per_level
        da_img_flattened = da_img_per_level
        da_img_labels_flattened = da_img_labels_flattened
        da_img_loss1 = F.binary_cross_entropy_with_logits(da_img_flattened, da_img_labels_flattened)
        return da_img_loss1
