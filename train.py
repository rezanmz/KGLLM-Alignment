import random
from torchkge.data_structures import KnowledgeGraph
import pandas as pd
from torchkge.sampling import BernoulliNegativeSampler
import torch
from torchkge import models
from torchkge.utils import MarginLoss
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import create_tsv_dataset

EMBEDDING_DIM = 512
LR = 0.001
EPOCHS = 2
MARGIN = 0.5
BATCH_SIZE = 16

TEXT_LOSS_EMPHASIS = 5
KG_LOSS_EMPHASIS = 2
SIMILARITY_LOSS_EMPHASIS = 2

# Normalize the emphasis of each loss
TEXT_LOSS_EMPHASIS = TEXT_LOSS_EMPHASIS / \
    (TEXT_LOSS_EMPHASIS + KG_LOSS_EMPHASIS + SIMILARITY_LOSS_EMPHASIS)
KG_LOSS_EMPHASIS = KG_LOSS_EMPHASIS / \
    (TEXT_LOSS_EMPHASIS + KG_LOSS_EMPHASIS + SIMILARITY_LOSS_EMPHASIS)
SIMILARITY_LOSS_EMPHASIS = SIMILARITY_LOSS_EMPHASIS / \
    (TEXT_LOSS_EMPHASIS + KG_LOSS_EMPHASIS + SIMILARITY_LOSS_EMPHASIS)

# Load data
triples, texts, mapping = create_tsv_dataset()
df_train = pd.read_csv('dataset/web_nlg_webnlg_challenge_2017_train_triples.tsv', sep='\t',
                       names=['from', 'rel', 'to'])
df_dev = pd.read_csv('dataset/web_nlg_webnlg_challenge_2017_dev_triples.tsv', sep='\t',
                     names=['from', 'rel', 'to'])
kg_train_dataset = KnowledgeGraph(df=df_train)
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available(
# ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
mapping['train']['triple'] = torch.tensor(
    mapping['train']['triple']).to(device)
mapping['train']['text'] = torch.tensor(mapping['train']['text']).to(device)
kg_negative_sampler = BernoulliNegativeSampler(kg_train_dataset)

kg_train_dataset.head_idx = kg_train_dataset.head_idx.to(device)
kg_train_dataset.relations = kg_train_dataset.relations.to(device)
kg_train_dataset.tail_idx = kg_train_dataset.tail_idx.to(device)

kg_model = models.TransEModel(EMBEDDING_DIM, kg_train_dataset.n_ent,
                              kg_train_dataset.n_rel, dissimilarity_type='L2').to(device)
kg_loss_fn = MarginLoss(margin=MARGIN).to(device)

text_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, mapping, tokenizer, max_length=128, device='cpu'):
        tokenized = tokenizer(text, padding=True, truncation=True, max_length=max_length,
                              return_tensors='pt').to(device)
        self.sentence_ids = torch.tensor(mapping['text']).to(device)
        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']

    def __len__(self):
        return self.sentence_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]


dataset = TextDataset(texts['train'], mapping['train'],
                      tokenizer, device=device)

optimizer = torch.optim.Adam(
    list(kg_model.parameters()) + list(text_model.parameters()), lr=LR)

pbar = tqdm(range(EPOCHS))

# Train
for epoch in pbar:
    # Batch
    n = max(mapping['train']['triple'])
    # Create a list and shuffle it
    sample_idx = list(range(n))
    random.shuffle(sample_idx)
    for i in range(0, n, BATCH_SIZE):
        # Get batch
        batch_idx = torch.tensor(sample_idx[i:i+BATCH_SIZE]).to(device)
        # Get triple idx
        triple_idx = torch.searchsorted(mapping['train']['triple'], batch_idx)
        # Get text idx
        text_idx = torch.searchsorted(mapping['train']['text'], batch_idx)
        # Get triples
        h = kg_train_dataset.head_idx[triple_idx].to(device)
        r = kg_train_dataset.relations[triple_idx].to(device)
        t = kg_train_dataset.tail_idx[triple_idx].to(device)

        # Get texts
        input_ids, attention_mask = dataset[text_idx]

        optimizer.zero_grad()
        # Train KG model
        n_h, n_t = kg_negative_sampler.corrupt_batch(h, t, r)
        pos, neg = kg_model(h, t, r, n_h, n_t)
        kg_loss = kg_loss_fn(pos, neg)
        kg_entity_embeddings, kg_relation_embeddings = kg_model.ent_emb.weight, kg_model.rel_emb.weight
        # Construct embedding for each triple
        triple_embeddings = (
            kg_entity_embeddings[h] + kg_relation_embeddings[r] + kg_entity_embeddings[t]) / 3

        # Train text model
        text_model_outputs = text_model(
            input_ids=input_ids,
            decoder_inputs_embeds=triple_embeddings.unsqueeze(
                1).expand(-1, dataset.input_ids.shape[1], -1),
            output_hidden_states=True
        )
        text_embeddings = torch.mean(
            text_model_outputs.encoder_last_hidden_state, dim=1)
        # reconstruction_loss = torch.nn.functional.mse_loss(
        #     text_embeddings, text_model_outputs.decoder_hidden_states[-1][:, 0, :])

        # Fine-tune text model
        text_loss = text_model(input_ids=input_ids, labels=input_ids).loss

        # Similarity loss
        similarity_loss = torch.nn.functional.cosine_embedding_loss(
            triple_embeddings, text_embeddings, torch.ones(triple_embeddings.shape[0]).to(device))

        # Total loss
        # loss = kg_loss + similarity_loss + reconstruction_loss
        loss = KG_LOSS_EMPHASIS * kg_loss \
            + SIMILARITY_LOSS_EMPHASIS * similarity_loss \
            + TEXT_LOSS_EMPHASIS * text_loss
        loss.backward()
        optimizer.step()

        pbar.set_description(
            f'Epoch {epoch} | Loss: {loss.item():.2f} | KG Loss: {kg_loss.item():.2f} | Text Loss: {text_loss.item():.2f} | Similarity Loss: {similarity_loss.item():.2f}')

# Save text model
text_model.save_pretrained('model/text_model')

# Save KG model
torch.save(kg_model.state_dict(), 'model/kg_model_with_text.pt')
