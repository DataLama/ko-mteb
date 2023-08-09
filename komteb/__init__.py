import os
from mteb import MTEB
from .tasks.Classification import KBSentiNeg, KlueYnat
from .tasks.PairClassification import QPair, KBBoolQ
from .tasks.STS import KlueSTS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_komteb_evaluation():
    """Initialize the ko-mteb evaluation."""
    evaluation = MTEB(task_langs=["ko"])
    return evaluation

