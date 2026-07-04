import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Optional, Sequence, Tuple, Union

_IMAGENET_MEAN_3 = (0.485, 0.456, 0.406)
_IMAGENET_STD_3 = (0.229, 0.224, 0.225)

_IMAGENET_MEAN_4 = (0.485, 0.456, 0.406, 0.406)
_IMAGENET_STD_4 = (0.229, 0.224, 0.225, 0.225)

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = int(self.mask_patch_size // self.model_patch_size)

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        """Return a patch-level mask aligned to model patch tokens.

        Output shape: [H_patches, W_patches] where H_patches = input_size / model_patch_size.
        Values are {0,1}.
        """
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int64)
        mask[mask_idx] = 1

        # rand_size x rand_size -> upsample to model patch grid via repetition
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask

class RingMoEDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6,
        modal_num: int = 1,
        modal_in_chans: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            data_path (str): Path to the json file containing image paths.
            input_size (int): Input image size.
            mask_patch_size (int): Patch size for masking.
            model_patch_size (int): Patch size for the model.
            mask_ratio (float): Masking ratio.
            modal_num (int): Number of modalities per sample. If >1, JSON must contain list-of-paths per item.
            modal_in_chans (Sequence[int], optional): Per-modality channel counts.
                - For modal_num==1, defaults to [3] if not provided.
                - For modal_num>1, must be provided and have length == modal_num.
        """
        if int(modal_num) < 1:
            raise ValueError(f"modal_num must be >= 1, got {modal_num}")
        self.data_dir = os.path.dirname(data_path)
        with open(data_path, 'r') as f:
            items = json.load(f)

        if not isinstance(items, list):
            raise ValueError("data json must be a list")

        self.modal_num = int(modal_num)
        if self.modal_num == 1:
            self.modal_in_chans = [int(modal_in_chans[0])] if modal_in_chans else [3]
        else:
            if modal_in_chans is None:
                raise ValueError("modal_in_chans must be provided when modal_num > 1")
            if len(modal_in_chans) != self.modal_num:
                raise ValueError(f"modal_in_chans must have length {self.modal_num}, got {len(modal_in_chans)}")
            self.modal_in_chans = [int(c) for c in modal_in_chans]
        self.items: List[Union[str, Sequence[str]]] = items

        # Normalize/validate JSON format.
        normalized: List[Union[str, List[str]]] = []
        for idx, item in enumerate(self.items):
            if self.modal_num == 1:
                if isinstance(item, str):
                    normalized.append(item)
                    continue
                if isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                    normalized.append(item[0])
                    continue
                raise ValueError(
                    f"data json item {idx} must be a string path when modal_num=1, got: {type(item)}"
                )

            if not (isinstance(item, list) and all(isinstance(p, str) for p in item)):
                raise ValueError(
                    f"data json item {idx} must be a list[str] when modal_num={self.modal_num}, got: {type(item)}"
                )
            if len(item) != self.modal_num:
                raise ValueError(
                    f"data json item {idx} has {len(item)} paths, expected modal_num={self.modal_num}"
                )
            normalized.append(list(item))

        def _resolve_path(p: str) -> str:
            return os.path.join(self.data_dir, p) if (p and not os.path.isabs(p)) else p

        # If paths are relative, join with data_dir
        if self.modal_num == 1:
            self.image_paths = [_resolve_path(p) for p in normalized]  # type: ignore[assignment]
        else:
            self.image_paths = [[_resolve_path(p) for p in item] for item in normalized]  # type: ignore[assignment]

        self.input_size = input_size
        self.model_patch_size = model_patch_size
        self.mask_generator = MaskGenerator(input_size, mask_patch_size, model_patch_size, mask_ratio)

    def _default_mean_std(self, in_chans: int) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
        if in_chans == 3:
            return _IMAGENET_MEAN_3, _IMAGENET_STD_3
        if in_chans == 4:
            return _IMAGENET_MEAN_4, _IMAGENET_STD_4
        return None, None

    def _load_array(self, path: str, expected_channels: int) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".npy", ".npz"):
            obj = np.load(path, allow_pickle=False)
            if isinstance(obj, np.lib.npyio.NpzFile):
                if "arr_0" in obj.files:
                    arr = obj["arr_0"]
                elif len(obj.files) == 1:
                    arr = obj[obj.files[0]]
                else:
                    raise ValueError(f"npz must contain a single array (or arr_0), got keys: {obj.files}")
            else:
                arr = obj
            return arr

        if ext in (".tif", ".tiff"):
            try:
                import tifffile as tf  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Reading .tif/.tiff requires tifffile. Install it or convert to .npy"
                ) from e
            return tf.imread(path)

        # Default: use PIL for common RGB/gray images.
        if expected_channels == 3:
            img = Image.open(path).convert("RGB")
            return np.asarray(img)
        if expected_channels == 1:
            img = Image.open(path).convert("L")
            return np.asarray(img)

        raise ValueError(
            f"Unsupported image extension/channel combo: path={path} expected_channels={expected_channels}. "
            "Use .npy/.npz (or .tif/.tiff with tifffile) for multi-channel modalities."
        )

    def _to_chw_float_tensor(self, arr: np.ndarray, expected_channels: int) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[..., None]  # HWC with C=1

        if arr.ndim != 3:
            raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")

        # Heuristic: accept HWC or CHW.
        if arr.shape[0] == expected_channels and arr.shape[2] != expected_channels:
            chw = arr  # CHW
        else:
            chw = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        if chw.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got array with shape {arr.shape} (converted CHW {chw.shape})"
            )

        t = torch.from_numpy(np.ascontiguousarray(chw))

        # Basic scaling for common integer formats.
        if t.dtype == torch.uint8:
            t = t.to(torch.float32) / 255.0
        elif t.dtype == torch.uint16:
            t = t.to(torch.float32) / 65535.0
        else:
            t = t.to(torch.float32)

        return t

    def _resize_and_normalize(self, x: torch.Tensor, expected_channels: int) -> torch.Tensor:
        # x: [C, H, W]
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)

        mean, std = self._default_mean_std(expected_channels)
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
            x = (x - mean_t) / std_t
        return x

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.modal_num == 1:
            img_path = self.image_paths[idx]
            in_chans = int(self.modal_in_chans[0])
            try:
                arr = self._load_array(img_path, expected_channels=in_chans)
                image = self._resize_and_normalize(self._to_chw_float_tensor(arr, in_chans), in_chans)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = torch.zeros(in_chans, self.input_size, self.input_size, dtype=torch.float32)

            mask = torch.from_numpy(self.mask_generator()).to(torch.int64)
            return image, mask

        img_paths = self.image_paths[idx]
        out = []
        for i, p in enumerate(img_paths):
            in_chans = int(self.modal_in_chans[i])
            try:
                arr = self._load_array(p, expected_channels=in_chans)
                img = self._resize_and_normalize(self._to_chw_float_tensor(arr, in_chans), in_chans)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                img = torch.zeros(in_chans, self.input_size, self.input_size, dtype=torch.float32)

            mask = torch.from_numpy(self.mask_generator()).to(torch.int64)
            out.append(img)
            out.append(mask)

        return tuple(out)
