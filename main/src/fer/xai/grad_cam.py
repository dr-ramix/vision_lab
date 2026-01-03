import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    """
    Grad-CAM implementation
    """
    def __init__(self, model, target_layers,reshape_transform=None):
        super(GradCAM,self).__init__(model,target_layers,reshape_transform)

    def get_cam_weights(self,input_tensor,target_layer,target_category,activations,grads):
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        else:
            raise ValueError("Invalid grads shape.")