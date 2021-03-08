import json
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset

from clip import tokenize

JPG_FORMAT = "COCO_{subset}_{image_id:012d}.jpg"


class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_dataset(image_preprocess, subset, images_path, questions_path, annotations_path=None, gt_threshold=3):
    """

    Parameters
    ----------
    image_preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor
        that BaseCLIP models can take as input
    subset : str
        used to fill the placeholder in JPG_FORMAT
        should be one of ["train2014", "val2014", "test2015"]
    images_path : PathLike
    questions_path : PathLike
    annotations_path : PathLike, optional
        If not provided, the text will consist only of the question (default).
    gt_threshold : int, optional
        number of annotators required to consider that the most common answer is ground-truth
        The QA pair will be discarded
        Defaults to 3.

    Returns
    -------
    dataset: VQADataset
    """
    # load annotation and question JSON files
    with open(questions_path, 'r') as file:
        questions = json.load(file)

    if annotations_path is not None:
        with open(annotations_path, 'r') as file:
            annotations = json.load(file)
        assert len(questions['questions']) == len(annotations['annotations'])
        qas = zip(questions['questions'], annotations['annotations'])
    else:
        annotations = None
        qas = questions['questions']

    # zip question and answers, preprocess text and image
    data = []
    for qa in qas:
        if annotations is not None:
            question, annotation = qa
            assert question['question_id'] == annotation['question_id']
            assert question['image_id'] == annotation['image_id']
        else:
            question = qa
        image_path = images_path / JPG_FORMAT.format(subset=subset, image_id=question['image_id'])
        image = image_preprocess(Image.open(image_path))

        # remove all punctuations marks in the question except for the final one
        text = question['question'].strip().replace("?", "") + "?"
        if annotations is not None:
            # lowercase, strip whitespaces ans remove all punctuations marks in the answer
            answers = Counter(answer['answer'].lower().strip().replace("?", "") for answer in annotation['answers'])
            answer, count = answers.most_common(1)[0]
            # skip question if there is so little agreement between annotators that the most common answer is below threshold
            if count < gt_threshold:
                continue
            text += " " + answer

        text = tokenize(text)[0]
        data.append(dict(inp=text, tgt=text, image=image))

    dataset = VQADataset(data)
    return dataset


def get_datasets(**kwargs):
    """

    Returns
    -------
    datasets: Tuple[Dataset]
        (train_dataset, eval_dataset)
        Defaults to None if "train_paths" (resp. "eval_paths") is not in kwargs
    """
    train_paths = kwargs.pop("train_paths", None)
    if train_paths is not None:
        train_dataset = get_dataset(**train_paths, **kwargs)
    else:
        train_dataset = None
    eval_paths = kwargs.pop("eval_paths", None)
    if eval_paths is not None:
        eval_dataset = get_dataset(**eval_paths, **kwargs)
    else:
        eval_dataset = None
    return train_dataset, eval_dataset
