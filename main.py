import json
import logging
import os
import shutil
import sys
from typing import Dict, Union
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, BertForTokenClassification, BertTokenizerFast, AutoTokenizer, AutoModelForTokenClassification, XLNetForTokenClassification, XLNetTokenizerFast
from transformers import pipeline

from seqeval.metrics import classification_report as seqeval_clf_rpt

logging.basicConfig(level=logging.INFO)


class FiNER_BERT:
    def __init__(self):
        self.train_path = 'data/train.csv'
        self.val_path = 'data/val.csv'
        self.test_path = 'data/test.csv'

        # FiNER labels
        self.int2str = {0: "O", 1: "PER_B", 2: "PER_I", 3: "LOC_B", 4: "LOC_I", 5: "ORG_B", 6: "ORG_I"}
        self.str2int = {v: k for k, v in self.int2str.items()}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.language_model = 'roberta-large'
        self.language_model = 'bert-base-cased'
        #self.language_model = 'roberta-base'

        """
        Tokenizer
            do_lower_case=False: 
                - because we want to preserve the case of the tokens, which can be important for named entity recognition task
            do_basic_tokenize=True: 
                - because we want to use the basic tokenizer of RoBERTa, which is designed to handle the specific tokenization needs of the model.
            add_prefix_space=True 
                - This tokenizer will tokenize the same word differently if it's at the beginning of a sentence (no preceding white space) or not (has preceding white space)
                - So bc in the FiNER dataset, each string processed by tokenizer is already a single word, then we set add_space_prefix=True to make sure a word is always tokenized the same (whether it's at the beginning of a sentence or not) and to avoid tokenizer treating each word (string) as the beginning of a sentence. As a result also we have to indicate is_split_into_words=True when calling it's forward
        """
        # self.tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-large', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        if self.language_model == 'bert-base-cased':
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'roberta-base':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'SALT-NLP/FLANG-Roberta':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('SALT-NLP/FLANG-Roberta', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'finbert-cased':# https://github.com/yya518/FinBERT
            self.tokenizer = BertTokenizerFast(vocab_file='../finbert-cased/FinVocab-Cased.txt', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'SALT-NLP/FLANG-BERT':
            self.tokenizer = BertTokenizerFast.from_pretrained('SALT-NLP/FLANG-BERT', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'bert-large-cased':
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'roberta-large':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)
        elif self.language_model == 'xlnet-base-cased':
            self.tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased", do_lower_case=False, do_basic_tokenize=True, add_prefix_space=True)


        # in load_data() when tokenizing, below flag indicates whether for a single word, to consider all of its subword tokens' NER predictions in the CE loss or only account for the first subword token's NER prediction when backpropagating the CE loss
        self.label_all_tokens = False # if True, all tokens for a single word will be assigned the same label and considered in the CE loss backprop, otherwise only the first token of a word will be assigned the label and the rest will be assigned -100 which will be ignored by the CE loss function during training since CE's ignore_index is set to -100 by default

        self.seeds = [5768, 78516, 944601]
        self.learning_rates = [1e-4, 1e-5, 1e-6]
        #self.learning_rates = [1e-5, 1e-6, 1e-7]
        self.batch_sizes = [32, 16, 8]
        self.num_epochs = 5
        self.early_stopping_limit = 7
        self.epsilon = 1e-2

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.current_experiment_state: Dict[str, Union[int, float, None]] = {
            'seed': None,
            'learning_rate': None,
            'batch_size': None,
        }

        self.experiment_version = "2.0"
        self.experiment_name = "BERT"
        self.results_path = "./"
        self.dir_path = os.path.join(self.results_path, f"{self.experiment_name}_results_v{self.experiment_version}")
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)
        
        self.best_val_accuracy = float('-inf')
        self.best_val_f1 = float('-inf')
        self.best_val_ce = float('inf')



    def set_criterion(self, train_str2int: Dict[str, int]):
        num_labels: int = len(train_str2int)
        weights = torch.ones(num_labels).to(self.device) * 1 / (num_labels - 1)
        weights[train_str2int['O']] = 0.001
        weights[train_str2int['PER_B']] = 0.1353 - 0.001/6
        weights[train_str2int['PER_I']] = 0.0911 - 0.001/6
        weights[train_str2int['LOC_B']] = 0.1592 - 0.001/6
        weights[train_str2int['LOC_I']] = 0.0476 - 0.001/6
        weights[train_str2int['ORG_B']] = 0.3338 - 0.001/6
        weights[train_str2int['ORG_I']] = 0.2330 - 0.001/6


        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

        self.logger.info(f"Classes: {train_str2int}, Class weights: {weights}")

        return self.criterion
    
    def set_current_lr(self, lr: float):
        self.current_experiment_state['learning_rate'] = lr

    def set_current_seed(self, seed: int):
        self.current_experiment_state['seed'] = seed

    def set_current_batch_size(self, seed: int):
        self.current_experiment_state['batch_size'] = seed

    def get_current_lr(self) -> float:
        assert self.current_experiment_state['learning_rate'] is not None, \
            f"Learning rate not set for the experiment yet"
        return self.current_experiment_state['learning_rate']

    def get_current_batch_size(self) -> float:
        assert self.current_experiment_state['batch_size'] is not None, \
            f"Batch size not set for the experiment yet"
        return self.current_experiment_state['batch_size']

    def get_current_seed(self) -> int:
        assert self.current_experiment_state['seed'] is not None, \
            f"Seed not set for the experiment yet"
        return self.current_experiment_state['seed']

    def get_criterion(self):
        assert self.criterion is not None, f"Criterion not set yet"
        return self.criterion
    


    def load_data(self, path, int2str: Dict[int, str]):
        df = pd.read_csv(path)
        # print(df.head())
        df.dropna(inplace=True)
        df.gold_label = df.gold_label.map(int2str) # turn the integer labels to string labels
        print(df.head())

        print("int2str mapping before sorting in load_data: ", int2str)

        df_sentences = df.groupby(['doc_idx', 'sent_idx']).agg({'gold_token': list, 'gold_label': list})
        sentences = df_sentences.gold_token.tolist()
        sentences_tags = df_sentences.gold_label.tolist()
        # print(sentences[0])
        # print(sentences_tags[0])
        # print(df_sentences)

        max_length = 0
        dropped_sentences = 0
        filtered_sentences = []
        filtered_sentences_tags = []
        for i, sentence in enumerate(sentences):
            try:
                # note even though the dataset is already tokenized by words, we still need to use the tokenizer to insert the special tokens such as [CLS] and [SEP] 
                # here we're going through and not including sentences that are too long for the model (greater than 512 tokens) because we want to avoid truncating them and losing important information, which can negatively impact the performance of the model. 
                # (note did not use padding or max_length params when tokenizing here because we just want to check the length of the tokenized sentence without adding any padding or truncation)
                tokens = self.tokenizer(sentence, is_split_into_words=True)
                sent_len = len(tokens['input_ids'])
                if sent_len <= 512:
                    filtered_sentences.append(sentence)
                    filtered_sentences_tags.append(sentences_tags[i])
                    max_length = max(max_length, sent_len)
                else:
                    dropped_sentences += 1
            except Exception as e:
                self.logger.error(f"Failed for sentence: {sentence} with exception: {e}")

        self.logger.info(f"Dropped {dropped_sentences} because of length greater than 512")
        sentences = filtered_sentences
        sentences_tags = filtered_sentences_tags

        # for some reason here the label's dictionary are reindexed by alphabetical order of the label's string values
        # why not just leave int_2_str as is?
        label_list = list(df.gold_label.unique())
        self.logger.info(f"Label list is: {label_list}")
        label_list.sort()
        print("Label list after sorting: ", label_list)
        str_to_int = {l: i for i, l in enumerate(label_list)}
        int_to_str = {i: l for (l, i) in str_to_int.items()}

        # str_to_int = {l: i for i, l in enumerate(int2str.values())}
        # int_to_str = int2str
        print("int_to_str mapping after sorting in load_data: ", int_to_str)

        tokenized_inputs = self.tokenizer(sentences,
                                            max_length=max_length,
                                            padding='max_length',
                                            is_split_into_words=True,
                                            return_tensors='pt')
        input_ids = tokenized_inputs['input_ids']
        attention_masks = tokenized_inputs['attention_mask']
        labels = []
        temp_global_list_labels = set()
        for i, label in enumerate(sentences_tags): # loop over each sentence's labels, note a single sentence is a single sequence
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # get the word_ids for the current sentence (i) from tokenizer's output
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # a special token such as [CLS] or [SEP] (which were added by the tokenizer, for BERT's understanding) will have a word_idx of None, we want to ignore these tokens when assigning labels bc for the NER task the special tokens don't correspond to any named entity, so we assign them a label of -100 which will be ignored by the CE loss function during training since CE's ignore_index is set to -100 by default. 
                # Also padding tokens will have word_idx of None so will be given label of -100 to be ignored by CE loss function during training as well, which is what we want since we don't want padding tokens to contribute to the loss during training
                if word_idx is None: 
                    label_ids.append(-100) # for NER task, ignore special tokens 
                # elif, this token is the first token of a word (a subword) or could be the full word. We want to predict the NER label for one of these. A word could be tokenized into multiple subword tokens or could just be one token
                elif word_idx != previous_word_idx:
                    # NOTE "label" is the ith list of labels for the current sentence, so label[word_idx] gives us the label for the current word, and then we convert that label from string to integer using str_to_int
                    label_ids.append(str_to_int[label[word_idx]]) 
                else:
                    # else, this means that the current token is a subword token of the same word as the previous token (not the first subword token), so we assign it the same label as the previous token if self.label_all_tokens is True, otherwise we assign it -100 to ignore it during training since we only want to assign the label to the first token of the word and ignore the rest of the subword tokens
                    # basically bc label_all_tokens is False, for the CE loss function, we will only consider the NER prediction of the first subword token of a word and ignore the rest of the subword tokens, which is a common approach in NER tasks to avoid overcounting the same word multiple times due to subword tokenization
                    label_ids.append(str_to_int[label[word_idx]] if self.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        labels = torch.LongTensor(labels)
        # return TensorDataset(input_ids, attention_masks, labels, doc_indices_tensor, sent_indices_tensor), str_to_int, int_to_str
        return TensorDataset(input_ids, attention_masks, labels), str_to_int, int_to_str



    def fine_tune(self, model, optimizer, dataloaders_dict, train_str2int: Dict[str, int]):
        seed = self.get_current_seed()
        num_labels: int = len(train_str2int)
        criterion = self.get_criterion() # loss function (CE)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.best_val_f1 = 0.0
        early_stopping_count = 0

        start_fine_tuning = time()
        for _ in tqdm(range(self.num_epochs), desc="# Epochs"):
            if early_stopping_count >= self.early_stopping_limit:
                break
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    early_stopping_count += 1
                else:
                    model.eval()
                curr_total = 0
                curr_correct = 0
                curr_ce = 0
                actual = np.array([])
                pred = np.array([])
                for input_ids, attention_masks, labels in tqdm(dataloaders_dict[phase]):
                    input_ids = input_ids.to(self.device)
                    attention_masks = attention_masks.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)   
                        # only consider the tokens that are not padding (attention_mask=1) for the loss calculation, ignore the tokens that are padding (attention_mask=0). Reshape attention_masks to (batch_size*seq_len,) 
                        active_loss = attention_masks.view(-1) == 1 
                        
                        # reshape (batch_size, seq_len, num_labels) to (batch_size*seq_len, num_labels), have to combine batch and sequence dimensions for CE loss
                        logits = outputs.logits
                        active_logits = logits.view(-1, num_labels) 
                        
                        # make sure tokens that are padding (attention_mask=0) will be ignored in the CE loss calculation by assigning their labels to -100 which is the ignore_index for CE loss function, and only consider the tokens that are not padding (attention_mask=1) for the CE loss calculation by keeping their original labels. Reshape labels to (batch_size*seq_len,)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
                        )
                        loss = criterion(active_logits, active_labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            curr_pred = outputs.logits.argmax(dim=-1).detach().cpu().clone().numpy()
                            curr_actual = labels.detach().cpu().clone().numpy()
                            true_predictions = np.concatenate([
                                [p for (p, l) in zip(sentence_preds, sentence_labels) if
                                 l != -100]
                                for sentence_preds, sentence_labels in zip(curr_pred, curr_actual)
                            ])
                            true_labels = np.concatenate([
                                [l for (p, l) in zip(sentence_preds, sentence_labels) if
                                 l != -100]
                                for sentence_preds, sentence_labels in zip(curr_pred, curr_actual)
                            ])
                            curr_correct += np.sum(true_predictions == true_labels)
                            curr_total += len(true_predictions)

                            # NOTE the CE's reduction is 'mean' by default (and we combine the batch and sequence dimensions), so here  multiplying the average token CE loss by the batch_size will get you the average sentence CE loss for the batch (each sequence is a sentence)
                            curr_ce += loss.item() * input_ids.size(0)
                            actual = np.concatenate([actual, true_labels], axis=0)
                            pred = np.concatenate([pred, true_predictions], axis=0)
                if phase == 'val':
                    curr_accuracy = curr_correct / curr_total
                    curr_f1 = f1_score(actual, pred, average='weighted')
                    curr_ce = curr_ce / len(dataloaders_dict[phase])
                    if curr_f1 >= self.best_val_f1 + self.epsilon:
                        self.best_val_f1 = curr_f1
                        self.best_val_ce = curr_ce
                        self.best_val_accuracy = curr_accuracy
                        early_stopping_count = 0
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            }, 'best_model.pt')
                        # model.save_pretrained("./final_model")
                    self.logger.info(f"Val Cross Entropy: {curr_ce}")
                    self.logger.info(f"Val Accuracy: {curr_accuracy}")
                    self.logger.info(f"Val F1: {curr_f1}")
                    self.logger.info(f"Early Stopping Count: {early_stopping_count}")
        self.fine_tuning_time = (time() - start_fine_tuning)/60.0
        # classifier = pipeline("ner", model=model, tokenizer=self.tokenizer, device=0, framework="pt")

        # example = "My name is Sarah and I live in London" 
        # ner_results = classifier(example)
        # for token in ner_results:
        #     print(token["word"], token["entity"])



    def test(self, model, optimizer, dataloaders_dict, train_str2int: Dict[str, int]):
        # model = RobertaForTokenClassification.from_pretrained(self.language_model,
        #                                                                num_labels=7).to(self.device)
        # create local instance of the model with the best performance from fine tuning
        if self.language_model == 'bert-base-cased':
            model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=7).to(self.device)
        elif self.language_model == 'roberta-base':
            model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=7).to(self.device)
        elif self.language_model == 'SALT-NLP/FLANG-Roberta':
            model = RobertaForTokenClassification.from_pretrained('SALT-NLP/FLANG-Roberta', num_labels=7).to(self.device)
        elif self.language_model == 'finbert-cased':
            model = BertForTokenClassification.from_pretrained('../finbert-cased/model', num_labels=7).to(self.device)
        elif self.language_model == 'SALT-NLP/FLANG-BERT':
            model = BertForTokenClassification.from_pretrained('SALT-NLP/FLANG-BERT', num_labels=7).to(self.device)
        elif self.language_model == 'bert-large-cased':
            model = BertForTokenClassification.from_pretrained('bert-large-cased', num_labels=7).to(self.device)
        elif self.language_model == 'roberta-large':
            model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=7).to(self.device)
        elif self.language_model == 'xlnet-base-cased':
            model = XLNetForTokenClassification.from_pretrained("xlnet-base-cased", num_labels=7).to(self.device)


        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        lr = self.get_current_lr()
        seed = self.get_current_seed()
        bs = self.get_current_batch_size()

        criterion = self.get_criterion()

        start_test_labeling = time()

        num_labels = len(train_str2int)
        test_total = 0
        test_correct = 0
        test_ce = 0
        actual = np.array([])
        pred = np.array([])

        y_true_entity_level_eval = []
        y_pred_entity_level_eval = []
        # mapping_dict = {
        #     'O': 'O',
        #     'PER_B': 'B-PER',
        #     'LOC_B': 'B-LOC',
        #     'PER_I': 'I-PER',
        #     'LOC_I': 'I-LOC',
        #     'ORG_B': 'B-ORG',
        #     'ORG_I': 'I-ORG'
        # }
        # mapping_dict = {
        #     0: 'O',
        #     1: 'B-PER',
        #     3: 'B-LOC',
        #     2: 'I-PER',
        #     4: 'I-LOC',
        #     5: 'B-ORG',
        #     6: 'I-ORG'
        # }

        # This is the mapping after changing the dictionary mapping by alphabetical order in load_data()
        # int2str mapping before sorting in load_data:  {0: 'O', 1: 'PER_B', 2: 'PER_I', 3: 'LOC_B', 4: 'LOC_I', 5: 'ORG_B', 6: 'ORG_I'}
        # int_to_str mapping after sorting in load_data:  {0: 'LOC_B', 1: 'LOC_I', 2: 'O', 3: 'ORG_B', 4: 'ORG_I', 5: 'PER_B', 6: 'PER_I'}
        mapping_dict = {0: 'B-LOC', 1: 'I-LOC', 2: 'O', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-PER', 6: 'I-PER'}

        # res_list = [] # list of list containing [doc_idx, sent_idx, true_label, pred_label]
        # seqeval_raw_map = {'doc_idx': [], 'sent_idx': [], 'true_label': [], 'pred_label': []}
        # seqeval_raw_map = {'sent_idx': [], 'true_label': [], 'pred_label': []}

        for input_ids, attention_masks, labels in dataloaders_dict['test']:
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_masks)
                active_loss = attention_masks.view(-1) == 1
                logits = outputs.logits
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
                )
                loss = criterion(active_logits, active_labels)
                curr_pred = outputs.logits.argmax(dim=-1).detach().cpu().clone().numpy()
                curr_actual = labels.detach().cpu().clone().numpy()
                # print(curr_pred[0])
                # print(curr_actual)
                # print(len(curr_pred))
                # print(len(curr_actual))
                true_predictions = np.concatenate([
                    [p for (p, l) in zip(sentence_preds, sentence_labels) if l != -100]
                    for sentence_preds, sentence_labels in zip(curr_pred, curr_actual)
                ])
                true_labels = np.concatenate([
                    [l for (p, l) in zip(sentence_preds, sentence_labels) if l != -100]
                    for sentence_preds, sentence_labels in zip(curr_pred, curr_actual)
                ])

                # sent_true_preds = []
                # sent_true_labels =[]
                for sentence_preds, sentence_labels in zip(curr_pred, curr_actual): # batch
                    curr_pred_list = []
                    curr_label_list = []
                    for p, l in zip(sentence_preds, sentence_labels): # sequence (sentence)
                        if l != -100:
                            curr_pred_list.append(p)
                            curr_label_list.append(l)
                    # convert the integer labels to their string NER entity labels using the mapping_dict
                    y_pred_entity_level_eval.append([mapping_dict[item] for item in curr_pred_list])
                    y_true_entity_level_eval.append([mapping_dict[item] for item in curr_label_list])
                # print(sent_true_preds)
                # print(sent_true_labels)

                test_total += len(true_predictions)
                test_ce += loss.item() * input_ids.size(0)
                test_correct += np.sum(true_predictions == true_labels)
                actual = np.concatenate([actual, true_labels], axis=0)
                pred = np.concatenate([pred, true_predictions], axis=0)

                # seqeval_raw_map['doc_idx'].extend(doc_indices_tensor.tolist())
                # seqeval_raw_map['sent_idx'].extend(sent_indices_tensor.tolist())
                # seqeval_raw_map['true_label'].extend(mapped_true_labels)
                # seqeval_raw_map['pred_label'].extend(mapped_pred_labels)
                # print(len(seqeval_raw_map['sent_idx']))
                # print(len(seqeval_raw_map['true_label']))
                # print(len(seqeval_raw_map['pred_label']))
                # break
            # break

                # print(true_labels)
                # print()
                # print(doc_indices_tensor)
                # print()
                # print(sent_indices_tensor)
                # print(mapped_true_labels)
                # print(true_predictions)
                # print(mapped_pred_labels)

                # y_true_entity_level_eval.append(mapped_true_labels)
                # y_pred_entity_level_eval.append(mapped_pred_labels)

                # print(y_true_entity_level_eval)
                # print(y_pred_entity_level_eval)
                # print()

                # break
        # print(y_true_entity_level_eval[30])
        test_time_taken = (time() - start_test_labeling)/60.0
        test_accuracy = test_correct / test_total
        test_ce = test_ce / len(dataloaders_dict['test'])
        test_f1 = f1_score(actual, pred, average='weighted')
        # print(seqeval_raw_map)
        # confusion_matrix_temp = confusion_matrix(actual,
        #                                pred,
        #                                labels=list(train_str2int.values()))
        # print(confusion_matrix_temp)

        #seqeval code for entity level metrics as requested in openreview
        # print(seqeval_clf_rpt(y_true_entity_level_eval, y_pred_entity_level_eval, digits=4))
        

        # report = classification_report(actual,
        #                                pred,
        #                                labels=list(train_str2int.values()),
        #                                target_names=list(train_str2int.keys()),
        #                                digits=4,
        #                                zero_division=0)

        report = seqeval_clf_rpt(y_true_entity_level_eval, y_pred_entity_level_eval, digits=4)
        print(report)
        # report_json = classification_report(actual,
        #                                     pred,
        #                                     labels=list(train_str2int.values()),
        #                                     target_names=list(train_str2int.keys()),
        #                                     digits=4,
        #                                     output_dict=True,
        #                                     zero_division=0)
        report_json = seqeval_clf_rpt(y_true_entity_level_eval, y_pred_entity_level_eval, digits=4,output_dict=True)

        report_filename = f"report_seed_{self.get_current_seed()}.csv"
        report_filepath = os.path.join(self.dir_path, report_filename)
        pd.DataFrame(report_json).to_csv(report_filepath)
        filename: str = os.path.join(self.dir_path, "results")
        header = not os.path.exists(f"{filename}.csv")
        pd.DataFrame([[seed, lr, bs, self.best_val_ce, self.best_val_accuracy, self.best_val_f1, test_ce, test_accuracy,
                       test_f1, self.fine_tuning_time, test_time_taken, report]],
                     columns=["Seed", "Learning Rate", "Batch Size", "Val CE", "Val Accuracy", "Val F1",
                              "Test CE", "Test Accuracy", "Test F1", "Fine Tuning Time(m)", "Test Labeling Time(m)", "classification_report"]).to_csv(
            f"{filename}.csv", mode='a', header=header)
    

    def grid_search(self):
        train_dataset, train_str2int, train_int2str = self.load_data(self.train_path, self.int2str)
        val_dataset, val_str2int, val_int2str = self.load_data(self.val_path, self.int2str)
        test_dataset, test_str2int, test_int2str = self.load_data(self.test_path, self.int2str)

        assert train_str2int == val_str2int == test_str2int, f"Labels are mismatching"
        assert train_int2str == val_int2str == test_int2str, f"Labels are mismatching"

        num_labels = len(train_int2str)
        self.set_criterion(train_str2int)

        # num_labels = len(test_dataset)
        # self.set_criterion(test_str2int)

        # main_seqeval_raw_map = {}

        # goes through and see which learning rate and batch size (with random seed) gives the best performance
        for seed in self.seeds:
            for lr in self.learning_rates:
                for bs in self.batch_sizes:
                    dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=bs, shuffle=True),
                                        'val': DataLoader(val_dataset, batch_size=bs, shuffle=True),
                                        'test': DataLoader(test_dataset, batch_size=bs, shuffle=False)} # could change shuffle=True, only order of sentences should be shuffled

                    self.set_current_seed(seed)
                    self.set_current_lr(lr)
                    self.set_current_batch_size(bs)

                    # model = RobertaForTokenClassification.from_pretrained(self.language_model,
                    #                                                    num_labels=num_labels).to(self.device)# , ignore_mismatched_sizes=True
                    if self.language_model == 'bert-base-cased':
                        model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=7).to(self.device)
                    elif self.language_model == 'roberta-base':
                        model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=7).to(self.device)
                    elif self.language_model == 'SALT-NLP/FLANG-Roberta':
                        model = RobertaForTokenClassification.from_pretrained('SALT-NLP/FLANG-Roberta', num_labels=7).to(self.device)
                    elif self.language_model == 'finbert-cased':
                        model = BertForTokenClassification.from_pretrained('../finbert-cased/model', num_labels=7).to(self.device)
                    elif self.language_model == 'SALT-NLP/FLANG-BERT':
                        model = BertForTokenClassification.from_pretrained('SALT-NLP/FLANG-BERT', num_labels=7).to(self.device)
                    elif self.language_model == 'bert-large-cased':
                        model = BertForTokenClassification.from_pretrained('bert-large-cased', num_labels=7).to(self.device)
                    elif self.language_model == 'roberta-large': # USING THIS MODEL
                        model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=7).to(self.device)
                    elif self.language_model == 'xlnet-base-cased':
                        model = XLNetForTokenClassification.from_pretrained("xlnet-base-cased", num_labels=7).to(self.device)
                    optimizer = optim.AdamW(model.parameters(), lr=lr)

                    self.fine_tune(model, optimizer, dataloaders_dict, train_str2int)
                    self.test(model, optimizer, dataloaders_dict, train_str2int)
                    # self.test(model, optimizer, dataloaders_dict, test_str2int)



if __name__ == "__main__":
    model = FiNER_BERT()

    model.grid_search()
    

