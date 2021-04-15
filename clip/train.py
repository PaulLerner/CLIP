"""Usage: train.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
import logging
from tqdm import tqdm
import collections

import numpy as np

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction, logging as t_logging
from transformers.trainer_callback import TrainerState
from transformers.file_utils import WEIGHTS_NAME
import torch
from torch import nn
from torch.autograd import set_detect_anomaly
from torch.utils.data.dataloader import DataLoader

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

from .clip import load, detokenize
import clip.model
from .data import get_datasets, collate_batch


logging.basicConfig()
logger = logging.getLogger(__name__)


class CLIPTrainer(Seq2SeqTrainer):
    """
    Like Seq2SeqTrainer but without the transformers specifics for generation
    Also passes all of the inputs to the model.generate method
    """

    def log(self, logs: Dict[str, float]) -> None:
        """Adds memory usage to the logs"""
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            logs[f"max_memory_{device}"] = torch.cuda.max_memory_allocated(device)
        return super().log(logs)

    def predict_and_save(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> List[Dict]:
        """
        Saves de-tokenized predictions along with question ID
        Adapted from prediction_loop (but doesn't compute metrics)
        Not implemented for distributed setting
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        answers = []
        for step, inputs in enumerate(dataloader):
            _, preds, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # tensor to numpy to List[str]
            preds = detokenize(preds.cpu().numpy(), answer_only=True, clean_up_tokenization_spaces=True)
            for pred, question_id in zip(preds, inputs['question_id']):
                answers.append(dict(answer=pred, question_id=question_id))
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        return answers

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
        """
        return self.model.generate(*args, **kwargs)


class PreTrainer(Learner):
    """
    Pre-trains CLIP using contrastive objective as described in Radford et al. 2021

    criterion should be CrossEntropyLoss as the model outputs raw logits
    """
    def forward(self, input_ids, *args, **kwargs):
        """Forward pass then compute the loss"""
        logits_per_image, logits_per_text = self.model(input_ids, *args, **kwargs)

        # compute the loss
        # image_i matches text_i
        labels = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
        loss_per_image = self.criterion(logits_per_image, labels)
        loss_per_text = self.criterion(logits_per_text, labels)
        loss = (loss_per_image + loss_per_text)/2

        return loss, (logits_per_image, logits_per_text)


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
    verbosity = config.get("verbosity", logging.INFO)
    t_logging.set_verbosity(verbosity)
    logger.setLevel(verbosity)

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
    trainer = CLIPTrainer(model=learner, args=training_args, data_collator=collate_batch,
                          train_dataset=train_dataset, eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics)
    return trainer, training_args, config


def get_checkpoint(resume_from_checkpoint: str, *args, **kwargs) -> List[Path]:
    if args or kwargs:
        logger.warning(f"ignoring additional arguments:\n{args}\n{kwargs}")
    cpt = Path(resume_from_checkpoint)
    # weird trick to glob using pathlib
    resume_from_checkpoints = list(cpt.parent.glob(cpt.name))
    return resume_from_checkpoints


def write_metrics(metrics, resume_from_checkpoint):
    logger.info(metrics)
    with open(resume_from_checkpoint/"metrics.json", "w") as file:
        json.dump(metrics, file)


def write_answers(answers, resume_from_checkpoint):
    with open(resume_from_checkpoint/"answers.json", "w") as file:
        json.dump(answers, file)


def main():
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = json.load(file)

    trainer, training_args, config = instantiate_trainer(config)
    checkpoint = config.get("checkpoint", {})
    if training_args.do_train:
        trainer.train(**checkpoint)
    elif training_args.do_eval:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Evaluation"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path)
            trainer.model.load_state_dict(state_dict)

            # optionally load trainer state for better logging
            trainer_state = resume_from_checkpoint/"trainer_state.json"
            if trainer_state.is_file():
                trainer.state = TrainerState.load_from_json(trainer_state)
            else:
                logger.warning("couldn't load trainer state, TB logging might use an inappropriate step")
            metrics = trainer.evaluate()
            write_metrics(metrics, resume_from_checkpoint)
    elif training_args.do_predict:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path)
            trainer.model.load_state_dict(state_dict)

            # run model on evaluation dataset
            eval_dataloader = trainer.get_eval_dataloader()
            answers = trainer.predict_and_save(eval_dataloader, description="Prediction")
            write_answers(answers, resume_from_checkpoint)
    else:
        logger.warning("Did nothing except instantiate the trainer, "
                       "you probably want to set do_train, do_eval or do_predict to True"
                       f"see {training_args.__doc__}")


if __name__ == "__main__":
    main()
