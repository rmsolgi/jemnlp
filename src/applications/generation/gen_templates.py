import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import Counter
####################################################################################
####################################################################################
####################################################################################
####################################################################################
class GenTaskTemplate(ABC):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.seq_len_training = None
        # self.seq_len_encode = None
        self.dataset = None  
        # self.task_type = None
        self.iterate_key = None
        self.columns_to_be_removed=None
        self.seq_len_filter_max=None
        self.QA=False
        self.generation = True
####################################################################################
    @abstractmethod
    def get_base_prompt(self, example):
        """Return the base prompt structure (varies by dataset)."""
        pass

    # @abstractmethod
    # def get_answer(self, example):
    #     """Return the correct candidate (varies by dataset)."""
    #     pass

    # @abstractmethod
    # def get_full_prompt(self, base_prompt, candid):
    #     """Return the full prompt including candidate (varies by dataset)."""
    #     pass

    def get_answer(self, example):
        """Default implementation: Subclasses can override if needed"""
        raise ValueError(" get_answer need to be override")
        return None  

    def get_full_prompt(self, base_prompt, candid=None):
        """Default implementation: Subclasses can override if needed"""
        raise ValueError(" get_full_prompt need to be override")
        return base_prompt
####################################################################################
    def instantiate(self):
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.dataset['validation']#.select(range(128))
    
####################################################################################
    def get_tokenized(self, inputs, mode):
        """Tokenize input based on mode (training or encoding)."""
        if mode == "training":
            return self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=self.seq_len_training,
                return_tensors="pt",
            )
        elif mode == "encode":
            return self.tokenizer.encode(
                inputs,
                truncation=False,
                padding=False,
                # max_length=self.seq_len_encode,
                return_tensors="pt",
            )
####################################################################################
    def get_tokenized_w_wo_candidate(self, base_prompt, candidate):
        """Tokenize base prompt and full prompt, computing candidate length."""
        full_prompt = self.get_full_prompt(base_prompt, candidate)
        full_tokenized = self.get_tokenized(full_prompt, mode="encode")
        base_tokenized = self.get_tokenized(base_prompt, mode="encode")

        full_len = full_tokenized.shape[1]
        base_len = base_tokenized.shape[1]

        if full_len > base_len:
            candid_len = full_len - base_len
        else:
            print("base: ", base_prompt)
            print("candid: ", candidate)
            print("full: ", full_prompt)
            print("full_t_len: ", full_len)
            print("base_t_len:", base_len)
            raise ValueError("choice length is not positive")

        return {
            "full_tokenized": full_tokenized,
            "base_tokenized": base_tokenized,
            "candid_len": candid_len,
        }
####################################################################################
    def get_data_stat(self):
        lengths = []
        # Iterate over each training example
        for example in self.dataset["train"]:
            # Get the prompt from the example
            # prompt = self.get_base_prompt(example)
            prompt=self.get_base_prompt(example)
            # Tokenize the prompt (ensure you're passing the prompt, not the entire example)
            tokenized = self.get_tokenized(prompt, mode='encode')
            # Record the length of the tokenized output
            # lengths.append(len(tokenized[0]))
            lengths.append(tokenized.shape[1])
        
        # Compute maximum tokenized length
        max_length = max(lengths)
        # Compute the 95th percentile length
        percentile_95 = np.percentile(lengths, 80)
        
        # Return the statistics as a dictionary
        return {"max_length": max_length, "80_percentile_length": percentile_95}
####################################################################################
    # def get_prompt_with_answer(self,example):
    #     prompt=self.get_base_prompt(example)
    #     # if self.QA:
    #     #     candid=self.get_answer(example)
    #     #     prompt=self.get_full_prompt(prompt,candid)
    #     return prompt
####################################################################################
    def print_example(self):
        counter=0
        for example in self.dataset['train']:
            prompt=self.get_base_prompt(example)
            if counter==0:
                print(prompt)
####################################################################################               
    def filter_long_prompts(self, example):
            prompt = self.get_base_prompt(example)
            tokenized = self.get_tokenized(prompt, mode='encode')
            # Check if the tokenized prompt is not too long
            return tokenized.shape[1] < self.seq_len_filter_max  # Assuming self.max_length is defined
 
 
####################################################################################
    def qa_preprocess_function(self, examples):
        """Preprocess dataset examples for tokenization and training."""
        inputs = [
            self.get_base_prompt({key: examples[key][i] for key in examples})
            + self.get_answer({key: examples[key][i] for key in examples})
            for i in range(len(examples[self.iterate_key]))
        ]

        tokenized = self.get_tokenized(inputs, mode="training")
        tokenized["labels"] = tokenized["input_ids"].clone()

        for i in range(len(tokenized["labels"])):
            example = {key: examples[key][i] for key in examples}
            base_prompt = self.get_base_prompt(example)
            correct_candidate = self.get_answer(example)
            answer_length = self.get_tokenized_w_wo_candidate(base_prompt, correct_candidate)["candid_len"]

            tokenized["labels"][i, :-answer_length] = -100  # Mask everything except the correct answer
        return tokenized
    
    def s2s_preprocess_function(self, examples):
        """Preprocess dataset examples for tokenization and training."""
        inputs = [
            self.get_base_prompt({key: examples[key][i] for key in examples})
            for i in range(len(examples[self.iterate_key]))
        ]

        tokenized = self.get_tokenized(inputs, mode="training")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized


    # def s2s_preprocess_function(self, examples, mask_ratio=0.8):
    #     # Generate input sequences
    #     inputs = [
    #         self.get_base_prompt({key: examples[key][i] for key in examples})  
    #         for i in range(len(examples[self.iterate_key]))
    #     ]

    #     # Tokenize inputs
    #     tokenized = self.get_tokenized(inputs, mode="training")

    #     # Clone input_ids as labels
    #     tokenized["labels"] = tokenized["input_ids"].clone()

    #     # Iterate over batch examples
    #     for i in range(len(tokenized["labels"])):
    #         # Create a mask to randomly set 80% of labels to -100
    #         prob_mask = torch.rand(tokenized["labels"].shape[1]) < mask_ratio  # Boolean mask (per token)
    #         tokenized["labels"][i, prob_mask] = -100  # Apply masking

    #     return tokenized

####################################################################################
    def get_train_dataset(self):
        """Tokenize the dataset and return train/eval splits."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Ensure the subclass defines `self.dataset`.")
        if self.QA is None:
            raise ValueError("QA must be set")

        if self.QA:
            tokenized_train_dataset = self.dataset["train"].map(
                self.qa_preprocess_function, batched=True, remove_columns=self.columns_to_be_removed
            )
        else:
            tokenized_train_dataset = self.dataset["train"].map(
                self.s2s_preprocess_function, batched=True, remove_columns=self.columns_to_be_removed
            )
        tokenized_train_dataset.set_format("torch")

        train_dataset = tokenized_train_dataset
        # eval_dataset = self.dataset["validation"]

        return train_dataset#, eval_dataset
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

class GenTaskTemplateMC(GenTaskTemplate): #for MULTIPLE CHOICE and CLASSIFICATION as generative tasks

    def __init__(self,tokenizer):
        super().__init__(tokenizer)
        self.QA=True

    @abstractmethod
    def get_all_candidates(self, example):
        """Return all possible candidates (varies by dataset)."""
        pass

####################################################################################
    def get_prompts_lens_and_label_for_eval(self, example):
        """Compute tokenized prompts and correct label for evaluation."""
        tokenized_prompts = []
        base_prompt = self.get_base_prompt(example)
        all_candidates = self.get_all_candidates(example)
        correct_candidate = self.get_answer(example)

        correct_label = None
        for i, candid in enumerate(all_candidates):
            tokenized = self.get_tokenized_w_wo_candidate(base_prompt, candid)
            candid_len = tokenized["candid_len"]
            full_prompt = self.get_full_prompt(base_prompt, candid)

            if candid_len > 0:
                tokenized_prompts.append({"candid_len": candid_len, "prompt": full_prompt})
            else:

                raise ValueError("choice length is not positive")

            if candid == correct_candidate:
                if correct_label is None:
                    correct_label = i
                else:
                    raise ValueError("Two correct candidates found.")

        if correct_label is None:
            raise ValueError("Correct candidate not found.")

        return tokenized_prompts, correct_label
####################################################################################

    def evaluate(self,model):
        """
        - Computes the average log-likelihood for each choice.
        - Selects the choice with the highest log-likelihood.
        - Reports accuracy and F1 score.
        """
        correct = 0
        total = 0
        device = model.device  # Get the device of the model (CPU or GPU)

        # Lists to store predictions and true labels for F1 computation
        predicted_labels_list = []
        true_labels_list = []

        def compute_log_likelihood(tokenized_prompt_and_len):
            """
            Computes the log-likelihood of a choice given a prompt.
            """
            input_text = tokenized_prompt_and_len['prompt']  # The full input prompt
            tokenized = self.tokenizer(input_text, return_tensors="pt", truncation=False)#, max_length=self.seq_len_encode)

            # Move input tensors to the same device as the model
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Compute log probabilities
            shift_logits = logits[:, :-1, :]  # Shift for prediction alignment
            shift_labels = input_ids[:, 1:]     # Next-token labels
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather the log probabilities of the actual tokens
            token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Compute average log-likelihood per token
            selected_probs=token_log_probs[-tokenized_prompt_and_len["candid_len"]:]
            # avg_log_likelihood = token_log_probs.mean().item()
            avg_log_likelihood = selected_probs.mean().item()

            return avg_log_likelihood

        # Iterate through each evaluation example
        for example in tqdm(self.eval_dataset, desc="Evaluation"):
            # bench.get_prompt_and_label(example) must return a list of prompts (one for each choice)
            # and the correct label as an integer.
            tokenized_prompts_and_lens, correct_label  = self.get_prompts_lens_and_label_for_eval(example)

            # Compute log-likelihood for each prompt dynamically
            log_likelihoods = torch.tensor([compute_log_likelihood(entry) for entry in tokenized_prompts_and_lens], device=device)

            # Get the predicted label index (i.e., index of maximum log-likelihood)
            predicted_label = log_likelihoods.argmax().item()

            if predicted_label == correct_label:
                correct += 1

            total += 1

            predicted_labels_list.append(predicted_label)
            true_labels_list.append(correct_label)

        accuracy = correct / total
        f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')

        return {"accuracy": accuracy, "f1_score (Macro)": f1}
####################################################################################
####################################################################################
####################################################################################
####################################################################################

class GenTaskTemplateS2S(GenTaskTemplate): #for Question Answering as generative tasks

    def __init__(self,tokenizer):
        super().__init__(tokenizer)
        self.QA=False

    def get_prompt_and_label_for_eval(self, example):
        """Compute tokenized prompts and correct label for evaluation."""
        # tokenized_prompts = []
        base_prompt = self.get_base_prompt(example)
        # all_candidates = self.get_all_candidates(example)
        tokenized = self.get_tokenized(base_prompt, mode="encode")
        labels = tokenized.clone()
        return tokenized, labels
       

    def evaluate(self, model):
        """Computes perplexity for the evaluation dataset with shifted labels."""
        all_log_probs = []
        total_tokens = 0

        # Ensure model is on CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # Move model to GPU if available
        
        for example in tqdm(self.eval_dataset, desc="Evaluation"):
            input_ids, labels = self.get_prompt_and_label_for_eval(example)

            # Move input tensors to the same device as the model
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

            # **Shift logits & labels for correct alignment**
            shift_logits = logits[:, :-1, :]  # Shift left to predict next token
            shift_labels = input_ids[:, 1:].to(device)   # Ensure labels are on same device

            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather log probabilities of correct next tokens
            token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Accumulate log probabilities and token count
            all_log_probs.append(token_log_probs.sum().detach())  # Keep as tensor, no `.item()`
            total_tokens += shift_labels.numel()

        # Compute perplexity (PPL)
        avg_log_prob = sum(all_log_probs) / total_tokens  # Keep computation on same device
        perplexity = torch.exp(-avg_log_prob)  # PPL = exp(-avg_log_prob)

        return {"perplexity": perplexity.item()}  # Convert to Python scalar only at the end




    # def evaluate(self, model):
    #     """Compute perplexity for the evaluation dataset with shifted labels, no batching."""

    #     all_nll = []
    #     total_tokens = 0

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     model.eval()

    #     loss_fct = nn.CrossEntropyLoss(reduction='sum')  # sum to accumulate token log-likelihoods

    #     for example in tqdm(self.eval_dataset, desc="Evaluation"):
    #         input_ids, labels = self.get_prompt_and_label_for_eval(example)  # assume both are shape [1, seq_len]

    #         input_ids = input_ids.to(device)
    #         labels = labels.to(device)

    #         with torch.no_grad():
    #             outputs = model(input_ids, labels=labels)
    #             logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    #         # Shift logits and labels
    #         shift_logits = logits[:, :-1, :].contiguous()
    #         shift_labels = input_ids[:, 1:].contiguous()

    #         # Flatten for loss: shape [seq_len-1, vocab_size] and [seq_len-1]
    #         loss = loss_fct(
    #             shift_logits.view(-1, shift_logits.size(-1)),
    #             shift_labels.view(-1)
    #         )

    #         all_nll.append(loss.detach())
    #         total_tokens += shift_labels.numel()

    #     total_nll = torch.stack(all_nll).sum()
    #     avg_nll = total_nll / total_tokens
    #     perplexity = torch.exp(avg_nll)

    #     torch.cuda.empty_cache()
    #     return {"perplexity": perplexity.item()}


####################################################################################
####################################################################################
####################################################################################
####################################################################################





class GenTaskTemplateQA(GenTaskTemplate): #for Question Answering as generative tasks

    def __init__(self,tokenizer):
        super().__init__(tokenizer)
        self.QA=True

####################################################################################
    def get_prompt_and_label_for_eval(self, example):
        """Compute tokenized prompts and correct label for evaluation."""
        # tokenized_prompts = []
        base_prompt = self.get_base_prompt(example)
        # all_candidates = self.get_all_candidates(example)
        answer = self.get_answer(example)
        full_prompt = self.get_full_prompt(base_prompt, answer)

        tokenized = self.get_tokenized(full_prompt, mode="encode")
        labels = tokenized.clone()

        answer_length = self.get_tokenized_w_wo_candidate(base_prompt, answer)["candid_len"]
        if not answer_length>0: 
            raise ValueError("answer length is zero for eval")

        labels[0, :-answer_length] = -100

        return tokenized, labels


####################################################################################


    def compute_f1_qa(self, prediction_tokens, ground_truth_tokens):
        """Computes F1 score based on token overlap."""
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def evaluate(self, model):
        """Custom evaluation function that generates answers and computes F1 score."""
        
        all_f1s = []
        max_new_tokens=0
        for example in tqdm(self.eval_dataset, desc="Evaluation"):
            input_ids, labels = self.get_prompt_and_label_for_eval(example)
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            # print(input_ids.shape)
            # print(labels.shape)
            #input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()  # Convert input_ids to tensor
            #labels = torch.tensor(labels).unsqueeze(0).cuda()  # Convert labels to tensor
            # Keep only tokens where labels are NOT -100 (i.e., original input part)
            filtered_input_ids = input_ids[labels == -100].unsqueeze(0)  # Reshape to match batch size
            attention_mask = (filtered_input_ids != self.tokenizer.pad_token_id).long()
            # filtered_input_ids = input_ids.clone()
            # filtered_input_ids[labels == -100] = tokenizer.pad_token_id  
            # Determine number of tokens to generate (equal to number of masked tokens)
            num_gen_tokens = (labels != -100).sum().item()
            # Generate `num_gen_tokens` new tokens
            generated_ids = model.generate(
                filtered_input_ids, 
                max_new_tokens=num_gen_tokens,  # Generate only missing tokens
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                # do_sample=False,  # Set True for diverse sampling, False for greedy decoding
                # early_stopping=False  
            )
            # Convert generated tokens to a list (removing padding tokens if needed)
            prediction_tokens = generated_ids[0].tolist()[-(num_gen_tokens):]  # Take last `num_gen_tokens`
            ground_truth_tokens = input_ids[0].tolist()[-(num_gen_tokens):]  # Extract ground truth answer tokens
            predicted_text = self.tokenizer.decode(prediction_tokens, skip_special_tokens=True)
            label_text = self.tokenizer.decode(ground_truth_tokens, skip_special_tokens=True)
            # # Compute F1 score
            # print('predicted: ', predicted_text)
            # print('label: ', label_text)
            f1_score = self.compute_f1_qa(prediction_tokens, ground_truth_tokens)
            all_f1s.append(f1_score)

        # Compute average F1-score
        avg_f1 = sum(all_f1s) / len(all_f1s)
        return {"f1_score": avg_f1}
######################################################################################################

