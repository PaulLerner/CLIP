"""Usage: train.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version

import numpy as np

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction, logging
import torch
from torch import nn
from torch.autograd import set_detect_anomaly
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

from .clip import load, detokenize
import clip.model
from .data import get_datasets


logger = logging.get_logger(__name__)


class CLIPTrainer(Seq2SeqTrainer):
    """
    Like Seq2SeqTrainer but without the transformers specifics for generation
    Also passes all of the inputs to the model.generate method
    """
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        generated_tokens = self.model.generate(**inputs)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]

        return (loss, generated_tokens, labels)


class Learner(nn.Module):
    """Base class for all learner models.
    Should implement a forward function that returns loss between output and target as first argument
    see, e.g., https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    So it can be used with transformers.Trainer

    Parameters
    ----------
    model: torch.nn.Module
        model to make the actual forward pass before computing the loss
    criterion: torch.optim.Optimizer
        criterion used to compute the loss between input and target
    """

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, input_ids, labels, *args, **kwargs):
        """Additional arguments should be passed to self.model"""
        out = self.model(input_ids, *args, **kwargs)
        # Flatten and compute the loss
        loss = self.criterion(out.view(-1, out.size(-1)), labels.view(-1))
        return loss, out

    def generate(self, *args, **kwargs):
        """
        Pass all arguments to self.model.generate
        Applies argmax to the last dimension of the model output to save memory
        (see https://github.com/huggingface/transformers/issues/10722)

        Beware this might break your code as it's not the expected behaviour in transformers Trainer
        """
        return self.model.generate(*args, **kwargs).argmax(-1)


class LanguageModel(Learner):
    """Shifts output and target tokens before computing the loss"""

    def forward(self, input_ids, labels, *args, **kwargs):
        out = self.model(input_ids, *args, **kwargs)
        # Shift so that tokens < n predict n
        shifted_out = out[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        # Flatten and compute the loss
        loss = self.criterion(shifted_out.view(-1, shifted_out.size(-1)), shifted_labels.view(-1))
        return loss, out


def compute_metrics(pred_and_label):
    """Note that the model is not supposed to output an EvalPrediction object
    The prediction is built inside trainer.prediction_loop

    Parameters
    ----------
    pred_and_label: EvalPrediction
        A named tuple with predictions and label_ids attributes
        Note that predictions is the output of model.generate and has already been "argmaxed"
        This function would not work with the output of model.forward (prob. distribution over the tokens)

    Returns
    -------
    metrics: dict
        {str: float}
    """
    predictions = pred_and_label.predictions
    labels = pred_and_label.label_ids
    predictions = np.asarray(list(detokenize(predictions, answer_only=True)), dtype=str)
    labels = np.asarray(list(detokenize(labels, answer_only=True)), dtype=str)
    accuracy = (predictions == labels).mean()
    return dict(accuracy=accuracy)


def instantiate_trainer(config):
    logging.set_verbosity(config.get("verbosity", logging.INFO))

    # debug (see torch.autograd.detect_anomaly)
    set_detect_anomaly(bool(config.get("debug", False)))
    
    # model
    model_args = dict(name="ViT-B/32", jit=False, training=True, Class="CLIPDecoder")
    model_args.update(config.get("model", {}))
    model_args["Class"] = getattr(clip.model, model_args["Class"])
    logger.info(f"loading model from pre-trained CLIP {model_args}...")
    model, image_preprocess = load(**model_args)

    # data
    train_dataset, eval_dataset = get_datasets(image_preprocess=image_preprocess, **config.get("dataset", {}))

    # training
    criterion_args = config.get("criterion", {})
    # get criterion class (e.g. nn.NLLLoss) by name
    CriterionClass = getattr(nn, criterion_args.pop("Class", "NLLLoss"))
    criterion = CriterionClass(**criterion_args)
    learner_args = config.get("learner", {})
    LearnerClass = getattr(sys.modules[__name__], learner_args.pop("Class", "LanguageModel"))
    learner = LearnerClass(model, criterion)
    training_args = Seq2SeqTrainingArguments(**config.get("training", {}))
    trainer = CLIPTrainer(model=learner, args=training_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics)
    return trainer, config


def main():
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = json.load(file)

    trainer, config = instantiate_trainer(config)
    trainer.train(**config.get("checkpoint", {}))


if __name__ == "__main__":
    main()
