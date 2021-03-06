import hashlib
import os
import urllib
import warnings
from typing import Union, List, Tuple

import numpy as np

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model, CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer, EOT_STR, SOT_STR

__all__ = ["available_models", "load", "tokenize", "tokenize_qa", "detokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit=True, training=False, Class=CLIP, fp16=True, context_length=None, pretrained=True, **kwargs):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device], optional
        The device to put the loaded model
        Defaults to cuda if available

    jit : bool, optional
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    training : bool, optional
        Whether to return the model in training or eval mode. Defaults to eval

    Class : type, optional
        Class of the model. Defaults to CLIP.

    fp16 : bool, optional
        Whether to use mixed-precision (default). Has no effect on CPU.

    context_length: int, optional
        Defaults to the size of the pre-trained model (i.e. in state_dict)
        Not available with jit

    pretrained: bool, optional
        Whether to load weights from pre-trained model (default) or just the config
        jit always loads pre-trained weights.

    **kwargs: additional arguments are passed to build_model

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").train(training)
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(),
                            training=training, Class=Class, fp16=fp16,
                            context_length=context_length, pretrained=pretrained, **kwargs).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)
    elif Class != CLIP:
        raise NotImplementedError(f"Set 'jit=False' for other models than CLIP (got {Class})")
    elif context_length is not None:
        warnings.warn(f"Setting context_length={context_length} has no effect with jit, "
                      f"keeping pre-trained value {model.context_length}")
    elif not pretrained:
        raise NotImplementedError(f"Set 'jit=False' to train the model from scratch")

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU and wrt fp16
    if str(device) == "cpu" or not fp16:
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize_qa(question: str, answer: str, context_length: int = 77, mask_question: bool = False) -> Tuple[torch.LongTensor]:
    """
    Parameters
    ----------
    question, answer: str
        Both will be tokenized and concatenated in inp
    context_length: int, optional
        The context length to use; all CLIP models use 77 as the context length
    mask_question: bool, optional
        Whether to pad the question or not (default) in the target
        i.e. train the model to predict both the question and the answer or only the answer

    Returns
    -------
    inp : Tensor
        (context_length, )
    tgt : Tensor
        (context_length, )
    """
    sot_token = _tokenizer.encoder[SOT_STR]
    eot_token = _tokenizer.encoder[EOT_STR]
    question_tokens = [sot_token] + _tokenizer.encode(question)
    answer_tokens = _tokenizer.encode(answer) + [eot_token]
    all_tokens = question_tokens + answer_tokens
    inp = torch.zeros(context_length, dtype=torch.long)
    tgt = torch.full_like(inp, -100)

    if len(all_tokens) > context_length:
        raise RuntimeError(f"Input {question+answer} is too long for context length {context_length}")

    tokens_tensor = torch.tensor(all_tokens)
    inp[: len(all_tokens)] = tokens_tensor

    # keep only the answer in the target
    if mask_question:
        tgt[len(question_tokens): len(all_tokens)] = torch.tensor(answer_tokens)
    else:
        tgt[: len(all_tokens)] = tokens_tensor

    return inp, tgt


def tokenize(texts: Union[str, List[str]], context_length: int = 77, return_tgt: bool = False) -> Tuple[torch.LongTensor]:
    """
    Returns the tokenized representation of the input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int, optional
        The context length to use; all CLIP models use 77 as the context length

    return_tgt : bool, optional
        If True, will also return tgt, see below

    Returns
    -------
    inp : Tensor
        A two-dimensional tensor containing the resulting tokens,
        shape = [number of input strings, context_length]
    tgt : Tensor, optional
        Same as inp but padded with -100 instead of 0
        These will be ignored by NLLLoss in training
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder[SOT_STR]
    eot_token = _tokenizer.encoder[EOT_STR]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    inp = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    if return_tgt:
        tgt = torch.full_like(inp, -100)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        tokens_tensor = torch.tensor(tokens)
        inp[i, :len(tokens)] = tokens_tensor
        if return_tgt:
            tgt[i, :len(tokens)] = tokens_tensor

    if return_tgt:
        return inp, tgt
    return inp


def detokenize(inp: np.ndarray,
               remove_special_tokens: bool = True,
               answer_only: bool = False,
               clean_up_tokenization_spaces=False) -> List[str]:
    """
    Yields the string representation of the input tokens (one str per item Tensor in the batch)

    Parameters
    ----------
    inp : np.ndarray
        (batch_size, context_length)
    remove_special_tokens : bool, optional
        Whether to keep only what's in between SOT_STR and EOT_STR (default)
    answer_only : bool, optional
        overrides remove_special_tokens: keep only what's in between "?" and EOT_STR
        Defaults to False
    clean_up_tokenization_spaces: bool, optional
        Remove spaces added during tokenization, e.g. in abbreviations, before punctuation

    Yields
    ------
    text: str
    """
    # set all padding tokens to 0 (instead of -100, see tokenize)
    inp[inp<0] = 0

    for tokens in inp:
        text = _tokenizer.decode(tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        if answer_only or remove_special_tokens:
            if answer_only:
                start = text.find("?") + 1
            elif remove_special_tokens:
                start = text.find(SOT_STR)
                if start < 0:
                    start = 0
                else:
                    start += len(SOT_STR)
            end = text.find(EOT_STR)
            if end < 0:
                end = len(text)
            text = text[start: end]
        yield text.strip()
