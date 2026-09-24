"""
gradcam.py

Core Grad-CAM (Gradient-weighted Class Activation Mapping) engine.

Works with any CNN where you can point to a target convolutional
layer. Built and tested against timm's efficientnet_b0, hooking
`model.conv_head` (the last conv block before global pooling).
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Usage:
        cam_engine = GradCAM(model, model.conv_head)
        cam, logits, pred_idx = cam_engine.generate(input_tensor)
        cam_engine.remove_hooks()
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] has the same shape as the layer's output
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W) tensor, already on the model's device.
        target_class: int class index to explain. If None, uses the
                       model's own top prediction (argmax of logits).

        Returns:
            cam        : numpy array (h, w), values normalized to [0, 1]
                         (h, w matches the target layer's spatial size,
                         e.g. 7x7 for EfficientNet-B0 at 224x224 input)
            logits     : raw model output, shape (1, num_classes), detached
            target_class: the class index the CAM was computed for
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[:, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients      # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global-average-pool the gradients -> per-channel importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach(), target_class
