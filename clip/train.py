"""Usage: train.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import warnings

from transformers import Trainer, TrainingArguments, EvalPrediction
from torch import nn

from .clip import load
from .model import CLIPDecoder
from .data import get_datasets


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

    def forward(self, inp, tgt, *args, **kwargs):
        """Additional arguments should be passed to self.model"""
        out = self.model(inp, *args, **kwargs)
        # Flatten and compute the loss
        loss = self.criterion(out.view(-1, out.size(-1)), tgt.view(-1))
        return loss, out


class LanguageModel(Learner):
    """Shifts output and target tokens before computing the loss"""

    def forward(self, inp, tgt, *args, **kwargs):
        out = self.model(inp, *args, **kwargs)
        # Shift so that tokens < n predict n
        shifted_out = out[..., :-1, :].contiguous()
        shifted_tgt = tgt[..., 1:].contiguous()
        # Flatten and compute the loss
        loss = self.criterion(shifted_out.view(-1, shifted_out.size(-1)), shifted_tgt.view(-1))
        return loss, out


def compute_metrics(prediction):
    """Note that the model is note supposed to output an EvalPrediction object
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
    warnings.warn("No additional metrics have been implemented yet, returning an empty dict.")
    return {}


if __name__ == "__main__":
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = json.load(file)

    # data
    train_dataset, eval_dataset = get_datasets(**config.get("dataset", {}))

    # model
    model_args = dict(name="ViT-B/32", jit=False, training=True, Class=CLIPDecoder)
    model_args.update(config.get("model", {}))
    model = load(**model_args)

    # training
    criterion = nn.NLLLoss(**config.get("criterion", {}))
    learner = LanguageModel(model, criterion)
    training_args = TrainingArguments(**config.get("training", {}))
    trainer = Trainer(model=learner, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)
    trainer.train(**config.get("checkpoint", {}))
    trainer.evaluate()
