from datasets import load_dataset, DatasetDict, Dataset
from src.applications.generation.gen_templates import GenTaskTemplateMC, GenTaskTemplate, GenTaskTemplateQA, GenTaskTemplateS2S
import re


####################################################################################
####################################################################################

def get_gen_bench(tokenizer, name):

    class_name = f"{name.upper()}_Benchmark" 

    if class_name in globals():
            return globals()[class_name](tokenizer)  
    else:
            raise ValueError(f"Dataset class '{class_name}' is not recognized.")




####################################################################################
####################################################################################
#########################   WIKITEXT2  
####################################################################################
####################################################################################


class WIKITEXT_Benchmark(GenTaskTemplateS2S):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.seq_len_training = 4096
        self.iterate_key = 'text'
        self.raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        self.dataset = DatasetDict({
            "train": self.prepare_split("train"),
            "validation": self.prepare_split("validation"),
            "test": self.prepare_split("test")
        })
        self.columns_to_be_removed = ["text"]
        self.instantiate()

    def get_base_prompt(self, example):
        return f"{example['text']}"

    def prepare_split(self, split_name):
        # Step 1: Join all texts with newlines (or spaces, depending on desired continuity)
        full_text = "\n\n".join(self.raw_dataset[split_name]["text"])
        
        # Step 2: Tokenize full stream
        tokens = self.tokenizer(full_text, return_tensors="pt").input_ids[0]

        # Step 3: Chunk into seq_len_training-sized sequences
        chunks = []
        for i in range(0, len(tokens) - self.seq_len_training + 1, self.seq_len_training):
            chunk_ids = tokens[i:i+self.seq_len_training]
            decoded = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(decoded)

        # Step 4: Return a Dataset object with the decoded text chunks
        return Dataset.from_dict({"text": chunks})



####################################################################################
####################################################################################
#########################   Alpaca  
####################################################################################
####################################################################################

class ALPACA_Benchmark(GenTaskTemplateS2S):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.seq_len_training = 2048
        self.iterate_key = 'text'

        # Load full alpaca dataset
        raw = load_dataset("tatsu-lab/alpaca")["train"]

        # Remove unwanted columns (keep only 'text')
        raw = raw.remove_columns([col for col in ["instruction", "input", "output"] if col in raw.column_names])

        # 80/10/10 split
        ds = raw.train_test_split(test_size=0.2, seed=42)
        temp = ds.pop("test")
        temp = temp.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp["train"]
        ds["validation"] = temp["test"]
        self.raw_dataset = ds

        # Prepare the chunked dataset
        self.dataset = DatasetDict({
            split: self.prepare_split(split) for split in ["train", "validation", "test"]
        })

        self.columns_to_be_removed = ["text"]
        self.instantiate()

    def get_base_prompt(self, example):
        return f"{example['text']}"

    def prepare_split(self, split_name):
        full_text = "\n\n".join(self.raw_dataset[split_name]["text"])
        tokens = self.tokenizer(full_text, return_tensors="pt").input_ids[0]

        chunks = []
        for i in range(0, len(tokens) - self.seq_len_training + 1, self.seq_len_training):
            chunk_ids = tokens[i:i+self.seq_len_training]
            decoded = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(decoded)

        return Dataset.from_dict({"text": chunks})

