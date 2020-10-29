from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, models
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from utils import  read_vocab, read_annotation_file, read_dataset


def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.replace('+', '|').split("|")).intersection(set(golden_cui.replace('+', '|').split("|"))))>0)

def is_correct(meddra_code, candidates, topk=1):
    for candidate in candidates[:topk]:
        if check_label(candidate, meddra_code): return 1
    return 0


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--model_dir')
  parser.add_argument('--data_folder')
  parser.add_argument('--vocab')
  parser.add_argument('--k', type=int, default=5)
  args = parser.parse_args()

  word_embedding_model = models.BERT(args.model_dir)
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  ################
  entities = read_dataset(args.data_folder)
  ################
  entity_texts = [e['entity_text'].lower() for e in entities]
  labels = [e['label'] for e in entities]
  ##################
  vocab = read_vocab(args.vocab)
  codes = vocab.label.values

  vocab_embeddings = model.encode(vocab.text.str.lower().tolist(), batch_size=128, show_progress_bar=True)
  vocab_embeddings = np.vstack(vocab_embeddings)
  tree = KDTree(vocab_embeddings)
  entities_embeddings = model.encode(entity_texts,  batch_size=128, show_progress_bar=True)

  correct_top1 = []
  correct_top5 = []
  for label, entity_embedding in tqdm(zip(labels, entities_embeddings), total=len(labels)):
    prediction_dists, prediction_idx = tree.query([entity_embedding], k=20)
    predicted_codes = codes[prediction_idx[0]]
    prediction_dists = prediction_dists[0]
    correct_top1.append(is_correct(label, predicted_codes, topk=1))
    correct_top5.append(is_correct(label, predicted_codes, topk=5))

  print("Acc@1 is ", np.mean(correct_top1))
  print("Acc@5 is ", np.mean(correct_top5))
