from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class HookOutput:
    name: str
    tensor: torch.Tensor


class ActivationManager:
    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        self.cache: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self) -> None:
        self.cache.clear()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.cache[name] = tensor.detach()

    def pop(self, name: str) -> Optional[torch.Tensor]:
        return self.cache.pop(name, None)

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.cache.get(name)

    def remove_hooks(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.clear()


def register_vit_hooks(model: nn.Module, manager: ActivationManager) -> ActivationManager:
    vit = getattr(model, "model", model)

    def patch_hook(_, __, output):
        manager.add("Layer_00_PatchEmbed", output)

    def pos_hook(module, inputs):
        x = inputs[0]
        manager.add("Layer_00_PosEmbed", x)

    def block_maker(i):
        name = f"Layer_{i:02d}_Block{i:02d}"
        def block_hook(_, __, output):
            manager.add(name, output)
        return block_hook

    def norm_hook(_, __, output):
        manager.add("Layer_13_CLS_Final", output[:, 0] if output.ndim == 3 else output)

    manager.handles.append(vit.patch_embed.register_forward_hook(patch_hook))
    if hasattr(vit, "pos_drop"):
        manager.handles.append(vit.pos_drop.register_forward_pre_hook(pos_hook))
    for i, block in enumerate(vit.blocks, start=1):
        manager.handles.append(block.register_forward_hook(block_maker(i)))
    if hasattr(vit, "norm"):
        manager.handles.append(vit.norm.register_forward_hook(norm_hook))
    return manager
