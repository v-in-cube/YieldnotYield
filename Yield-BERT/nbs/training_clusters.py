from rxnfp.tokenization import get_default_tokenizer, SmilesTokenizer
from rdkit.Chem import rdChemReactions
import pandas as pd
smiles_tokenizer = get_default_tokenizer()

import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
)
yield_bert = SmilesClassificationModel('bert', model_path, use_cuda=torch.cuda.is_available())

# data
model_args = {
     'num_train_epochs': 30, 'overwrite_output_dir': True,
    'learning_rate': 0.00009659, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.4 } 
}

model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
)

yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, 
                                       args=model_args, use_cuda=torch.cuda.is_available())


df = pd.read_csv("df_hte_cluster_drfp_14.csv")

for cluster in df["Cluster"].unique():
    
    train = df[df["Cluster"]!=cluster]
    
    train_df=train.sample(frac=0.8,random_state=200)
    
    eval_df=train.drop(train_df.index)
    test = df[df["Cluster"]==cluster]
    
    train_df = train_df[["rxn", "YIELD"]]
    eval_df = eval_df[["rxn", "YIELD"]]
    test_df = test[["rxn", "YIELD"]]
    
    train_df.columns = ['text', 'labels']
    eval_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']
    
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    
    train_df['labels'] = (train_df['labels'] - mean) / std
    eval_df['labels'] = (eval_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std

    yield_bert.train_model(train_df, output_dir=f"outputs_buchwald_hartwig_cross_val_project_cluster_{cluster}", eval_df=eval_df)