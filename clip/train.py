"""Usage: train.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import sys

import numpy as np

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction, logging, PretrainedConfig
import torch
from torch import nn
from torch.autograd import set_detect_anomaly

from .clip import load, detokenize
import clip.model
from .data import get_datasets


logger = logging.get_logger(__name__)


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
        return self.model.generate(*args, **kwargs)


class LanguageModel(Learner):
    """Shifts output and target tokens before computing the loss"""

    def __init__(self, model, criterion, max_length=77):
        super().__init__(model, criterion)
        self.config = PretrainedConfig(max_length=max_length, num_beams=1)

    def forward(self, input_ids, labels, *args, **kwargs):
        out = self.model(input_ids, *args, **kwargs)
        # Shift so that tokens < n predict n
        shifted_out = out[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        # Flatten and compute the loss
        loss = self.criterion(shifted_out.view(-1, shifted_out.size(-1)), shifted_labels.view(-1))
        return loss, out


def compute_metrics(prediction):
    """Note that the model is not supposed to output an EvalPrediction object
    The prediction is built inside trainer.prediction_loop

    Parameters
    ----------
    prediction: EvalPrediction
        A named tuple with predictions and label_ids attributes
        Note that predictions is the output of the model and has not been "argmaxed"

    Returns
    -------
    metrics: dict
        {str: float}
    """
    predictions = prediction.predictions.argmax(-1)
    labels = predictions.label_ids
    predictions = np.asarray(detokenize(predictions, answer_only=True), dtype=str)
    labels = np.asarray(detokenize(labels, answer_only=True), dtype=str)
    accuracy = (predictions == labels).mean()
    return dict(accuracy=accuracy)


def main():
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = json.load(file)

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
    trainer = Seq2SeqTrainer(model=learner, args=training_args,
                             train_dataset=train_dataset, eval_dataset=eval_dataset,
                             compute_metrics=compute_metrics)
    trainer.train(**config.get("checkpoint", {}))


if __name__ == "__main__":
    main()
