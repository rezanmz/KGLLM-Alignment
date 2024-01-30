from torchkge import models
from torchkge.data_structures import KnowledgeGraph
import pandas as pd
import torch
from torchkge.evaluation import LinkPredictionEvaluator

device = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'

df_train = pd.read_csv('dataset/web_nlg_webnlg_challenge_2017_train_triples.tsv', sep='\t',
                       names=['from', 'rel', 'to'])
kg_train_dataset = KnowledgeGraph(df=df_train)

model_no_text = models.TransEModel(384, kg_train_dataset.n_ent,
                                   kg_train_dataset.n_rel, dissimilarity_type='L2').to(device)
model_no_text.load_state_dict(torch.load('model/kg_model_no_text.pt'))

print('KG model trained without using text:')
evaluator = LinkPredictionEvaluator(model_no_text, kg_train_dataset)
evaluator.evaluate(b_size=32)
evaluator.print_results()

model_with_text = models.TransEModel(384, kg_train_dataset.n_ent,
                                     kg_train_dataset.n_rel, dissimilarity_type='L2').to(device)
model_with_text.load_state_dict(torch.load('model/kg_model_with_text.pt'))

print('KG model trained using text:')
evaluator = LinkPredictionEvaluator(model_with_text, kg_train_dataset)
evaluator.evaluate(b_size=32)
evaluator.print_results()
