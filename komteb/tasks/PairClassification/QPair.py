import datasets
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification

class QPair(AbsTaskPairClassification):
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # hf_part is the name of load_dataset
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None) 
        )
        # convert for mteb pairclassification form.
        dict_of_list = {'sent1':self.dataset['test']['question1'], 'sent2':self.dataset['test']['question2'], 'labels':self.dataset['test']['label']}
        self.dataset = datasets.DatasetDict({'test': datasets.Dataset.from_list([dict_of_list])})
        self.data_loaded = True
        
    @property
    def description(self):
        return {
            "name": "QPair",
            "hf_hub_name": "datalama/question_pair",
            "description": "Question Pair Binary Classification",
            "reference": "https://github.com/songys/Question_pair",
            "type": "PairClassification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["ko"],
            "main_score": "ap" # average precision
        }