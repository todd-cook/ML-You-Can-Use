import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import logging

# +
from sklearn.preprocessing import LabelEncoder
import math


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class SearchRelevanceDataset(Dataset):
    """A dataset for search relevance.

    Typically expected feature columns:
    'query', 'product_description'

    label: median_relevance

    """

    def __init__(self, filename, maxlen,
                 tokenizer=None,
                 tokenizer_name='bert-base-uncased',
                 data_columns =None,
                 label_name='median_relevance',
                 max_len=512):
        """initializations"""
        self.label_name = label_name
        self.data_columns = data_columns 
        if not self.data_columns:
            self.data_columns = ('query', 'product_description') # product_title
        self.max_len=max_len
        self.data_df = pd.read_csv(filename, error_bad_lines=False)
        # correct empty items, it better not be necessary for all
        for col in data_columns:
            self.data_df[col ].fillna('', inplace=True)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            #Initialize the BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        if self.label_name:
            uniq_labels = self.data_df[label_name].unique()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(uniq_labels)

    def __len__(self):
        return len(self.data_df)

    def create_token_type_ids(self, tids):
        first_sep = tids.tolist().index(102) + 1
        return torch.cat((torch.zeros(first_sep, dtype=torch.long), torch.ones(len(tids)- first_sep, dtype=torch.long)))

    
    def __getitem__(self, index):
        """provide the data"""
        query = self.tokenizer.encode(str(self.data_df.loc[index, self.data_columns[0]]), add_special_tokens=False)
        title = self.tokenizer.encode(str(self.data_df.loc[index, self.data_columns[1]]), add_special_tokens=False)
        # max_len - 3 ; the 3 is needed to accommodate the special tokens
        description = self.tokenizer.encode(str(self.data_df.loc[index, self.data_columns[2]] )[:self.max_len - 4],
                                       max_length=self.max_len - 4,
                                       pad_to_max_length=True,
                                       add_special_tokens=False)

        token_ids = [101] + query + [102] + title + [102] + description [:self.max_len -4 - len(query) - len(title)] + [102]
        # Selecting the sentence and label at the specified index in the data frame
        if self.label_name is not None: 
            label_val = self.data_df.loc[index, self.label_name]
            label = self.label_encoder.transform([label_val])[0]
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        tokens_ids_tensor = torch.tensor(token_ids)  # Converting the list to a pytorch tensor
        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        token_type_ids = self.create_token_type_ids(tokens_ids_tensor)
        return tokens_ids_tensor, token_type_ids, attn_mask, label
