from transformers import RobertaModel, XLMRobertaConfig
from transformers import BertPreTrainedModel
from transformers import XLMRobertaTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
import torch, os, json
from collections import defaultdict
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


from scipy.special import softmax
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report, accuracy_score
)


def wsd_framework_score(golds, preds):
    ok, notok = 0, 0
    for target_id in preds:
        if target_id not in golds:
            continue
        local_ok, local_notok = 0,  0
        for sense_id in preds[target_id]:
            if sense_id in golds[target_id]:
                local_ok += 1
            else:
                local_notok += 1
        ok += (local_ok / len(preds[target_id]))
        notok += (local_notok / len(preds[target_id]))
    p = ok / (ok + notok)
    r = ok / len(golds)
    f = 0.0 if p + r == 0.0 else (2 * p * r) / (p + r)
    return p, r, f


XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                 "-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                  "-large-finetuned-conll03-german-pytorch_model.bin",
}


class WiCFeature2:
    def __init__(self, input_ids, input_mask, token_type_ids, syn_label, positions, example):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.syn_label = syn_label
        self.positions = positions
        self.example = example


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_classes, input_size, args):
        super().__init__()
        bn = args.head_batchnorm
        self.linear_head = args.linear_head
        hidden_size = args.head_hidden_size
        hidden_size = config.hidden_size if hidden_size <= 0 else hidden_size
        self.bn1 = torch.nn.BatchNorm1d(input_size) if bn % 2 == 1 else None
        self.bn2 = torch.nn.BatchNorm1d(hidden_size) if bn // 2 == 1 else None
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(input_size if self.linear_head else hidden_size, num_classes)
        print(f'RobertaClassificationHead: linear_head={self.linear_head}, hs={hidden_size}, input={input_size}, bn={bn}')

    def forward(self, features, **kwargs):
        x = features if self.bn1 is None else self.bn1(features)
        if not self.linear_head:
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = x if self.bn2 is None else self.bn2(x)
            x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLMRModel(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(
            self,
            config: XLMRobertaConfig,
            args,
            data_processor
        ):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.args = args
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
        input_size = config.hidden_size

        if args.pool_type in {'mmm','mmf'}:
            input_size *= 3
        elif args.pool_type in {'mm','mf'}:
            input_size *= 2

        if args.target_embeddings == 'concat':
            input_size *= 2
        elif args.target_embeddings.startswith('comb_c'):
            input_size *= 3
        elif args.target_embeddings.startswith('comb_'):
            input_size *= 2
        elif args.target_embeddings.startswith('dist_'):
            input_size = len(args.target_embeddings.replace('dist_', '').replace('n', '')) // 2

        print('Classification head input size:', input_size)
        if self.args.loss == 'mse_loss':
            self.syn_mse_clf = RobertaClassificationHead(config, 1, input_size, self.args)
        elif self.args.loss == 'crossentropy_loss':
            self.syn_clf = RobertaClassificationHead(config, 2, input_size, self.args)
        self.data_processor = data_processor
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_labels=None,
    ):
        loss = defaultdict(float)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        layers = sorted([int(x) for x in self.args.layers.split(',')], reverse=True)
        sequences_output = outputs[2][layers[0]]
        for layer in layers[1:]:
            sequences_output += outputs[2][layer] # bs x seq x hidden

        syn_labels = input_labels['syn_labels']  # bs
        if self.args.siamese:
            syn_labels_ids = [i for i in range(len(syn_labels)) if i % 2]  # bs // 2
            syn_labels = syn_labels[syn_labels_ids] # bs // 2
            # assert len(syn_labels) == sequences_output.shape[0] // 2
        positions = input_labels['positions'] # bs x 4

        syn_features = self.extract_features(sequences_output, positions) # bs x hidden or (bs // 2) x hidden
        mse_loss = self.args.loss == 'mse_loss'
        clf = self.syn_mse_clf if mse_loss else self.syn_clf
        syn_logits = clf(syn_features)

        if input_labels is not None:
            if mse_loss:
                loss['total'] = MSELoss()(syn_logits, syn_labels.unsqueeze(-1).float())
            elif self.args.loss == 'crossentropy_loss':
                loss['total'] = CrossEntropyLoss()(syn_logits, syn_labels)

        return (loss, syn_logits)

    def extract_features(self, hidden_states, positions):
        pool_type = self.args.pool_type
        merge_type = self.args.target_embeddings
        bs, seq, hs = hidden_states.size()
        features = []
        niters = bs // 2 if self.args.siamese else bs
        batch_positions = lambda x: (x * 2, x * 2 + 1) if self.args.siamese else (x, x)
        ex1_st, ex1_end, ex2_st, ex2_end = (0, 1, 0, 1) if self.args.siamese else  (0, 1, 2, 3)

        for ex_id in range(niters):
            ex1_id, ex2_id = batch_positions(ex_id)
            start1, end1, start2, end2 = positions[ex1_id, ex1_st].item(), positions[ex1_id, ex1_end].item(), positions[ex2_id, ex2_st].item(), positions[ex2_id, ex2_end].item()
            if pool_type == 'mean':
                emb1 = hidden_states[ex1_id, start1:end1].mean(dim=0) # hidden
                emb2 = hidden_states[ex2_id, start2:end2].mean(dim=0) # hidden
            elif pool_type == 'max':
                emb1, _ = hidden_states[ex1_id, start1:end1].max(dim=0) # hidden
                emb2, _ = hidden_states[ex2_id, start2:end2].max(dim=0) # hidden
            elif pool_type == 'first':
                emb1 = hidden_states[ex1_id, start1]
                emb2 = hidden_states[ex2_id, start2]
            elif pool_type == 'mf':
                embs1 = hidden_states[ex1_id, start1:end1] # hidden
                embs2 = hidden_states[ex2_id, start2:end2] # hidden
                emb1, emb2 = (torch.cat([embs.mean(dim=0), embs[0]], dim=-1) for embs in (embs1, embs2))
            elif pool_type.startswith('mm'):
                embs1 = hidden_states[ex1_id, start1:end1] # hidden
                embs2 = hidden_states[ex_id, start2:end2] # hidden
                last = '' if len(pool_type) < 3 else pool_type[2]
                emb1, emb2 = (torch.cat([embs.mean(dim=0), embs.max(dim=0).values] + ([] if last == '' else [embs[0]] if last == 'f' else [embs.min(dim=0).values]), dim=-1) for embs in (embs1, embs2))
            else:
                raise ValueError(f'wrong pool_type: {pool_type}')
            if merge_type == 'featwise_mul':
                merged_feature = emb1 * emb2 # hidden
            elif merge_type == 'diff':
                merged_feature = emb1 - emb2
            elif merge_type == 'absdiff':
                merged_feature = torch.abs(emb1 - emb2)
            elif merge_type == 'concat':
                merged_feature = torch.cat((emb1, emb2)) # 2 * hidden
            elif merge_type == 'mulnorm':
                merged_feature = (emb1 / emb1.norm(dim=-1, keepdim=True)) * (emb2 / emb2.norm(dim=-1, keepdim=True))
            elif merge_type == 'comb_cm':
                merged_feature = torch.cat((emb1, emb2, emb1 * emb2))
            elif merge_type == 'comb_cmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1, emb2, emb1n*emb2n))
            elif merge_type == 'comb_cd':
                merged_feature = torch.cat((emb1, emb2, emb1 - emb2))
            elif merge_type == 'comb_cnmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1n, emb2n, emb1n * emb2n))
            elif merge_type == 'comb_dmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1 - emb2, emb1n * emb2n))
            elif merge_type == 'comb_dnmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1n - emb2n, emb1n * emb2n))
            elif merge_type.startswith( 'dist_'):
                if 'n' in merge_type:
                    emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                    emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                dists = []
                if 'l1' in merge_type:
                    dists.append(torch.norm(emb1n-emb2n if 'l1n' in merge_type else emb1-emb2, dim=-1, p=1, keepdim=True))
                if 'l2' in merge_type:
                    dists.append(torch.norm(emb1n-emb2n if 'l2n' in merge_type else emb1-emb2, dim=-1, p=2, keepdim=True))
                if 'dot' in merge_type:
                    dists.append((emb1n * emb2n if 'dotn' in merge_type else emb1*emb2 ).sum(dim=-1, keepdim=True))
                merged_feature = torch.cat(dists)

            features.append(merged_feature.unsqueeze(0))
        output = torch.cat(features, dim=0) # bs x hidden or (bs // 2) x hidden
        return output

    def convert_dataset_to_features(
            self, source_dir, logger
            ):
        features = []
        max_seq_len = self.args.max_seq_len

        examples = self.data_processor.get_examples(source_dir)
        syn_label_to_id = {'T': 1, 'F': 0}
        num_too_long_exs = 0
        skipped = 0

        for (ex_index, ex) in enumerate(examples):
            pos, label = ex.pos, ex.label
            for i, (st1, end1, sent1, st2, end2, sent2) in enumerate(
                    [(ex.start_1, ex.end_1, ex.text_1, ex.start_2, ex.end_2, ex.text_2),
                     (ex.start_2, ex.end_2, ex.text_2, ex.start_1, ex.end_1, ex.text_1)]):
                if not self.args.symmetric and i != 0:
                    continue
                st1, end1, st2, end2 = int(st1), int(end1), int(st2), int(end2)
                tokens1 = [self.tokenizer.cls_token]
                tokens2 = [self.tokenizer.cls_token]

                positions1 = []
                positions2 = []
                left1, target1, right1 = sent1[:st1], sent1[st1:end1], sent1[end1:]
                left2, target2, right2 = sent2[:st2], sent2[st2:end2], sent2[end2:]

                if left1:
                    tokens1 += self.tokenizer.tokenize(left1)
                if left2:
                    tokens2 += self.tokenizer.tokenize(left2)

                positions1.append(len(tokens1))  # start of the first target
                positions2.append(len(tokens2))  # start of the second target

                target1_subtokens = self.tokenizer.tokenize(target1)
                target2_subtokens = self.tokenizer.tokenize(target2)
                if self.args.mask_targets:
                    tokens1 += [self.tokenizer.mask_token] * len(target1_subtokens)
                    tokens2 += [self.tokenizer.mask_token] * len(target2_subtokens)
                else:
                    tokens1 += target1_subtokens
                    tokens2 += target2_subtokens

                positions1.append(len(tokens1))  # end of the first target
                positions2.append(len(tokens2))  # end of the second target

                if right1:
                    tokens1 += self.tokenizer.tokenize(right1) + [self.tokenizer.sep_token]
                if right2:
                    tokens2 += self.tokenizer.tokenize(right2) + [self.tokenizer.sep_token]

                if self.args.siamese:
                    if max([position for positions in [positions1, positions2] for position in
                            positions]) > max_seq_len - 1:
                        num_too_long_exs += 2
                        skipped += 2
                        continue
                    splitted_or_not_example = [(tokens1, positions1), (tokens2, positions2)]
                else:
                    if max(positions2) - 1 + len(tokens1) > max_seq_len - 1:
                        num_too_long_exs += 1
                        skipped += 1
                        continue
                    positions1 += [position - 1 + len(tokens1) for position in positions2]
                    tokens1 += tokens2[1:]
                    assert max(positions1) <= len(tokens1), f'{(positions1, len(tokens1))}'
                    splitted_or_not_example = [(tokens1, positions1)]

                for trgt_id, (tokens, positions) in enumerate(splitted_or_not_example):
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    if len(input_ids) > max_seq_len:
                        input_ids = input_ids[:max_seq_len]
                        num_too_long_exs += 1

                    input_mask = [1] * len(input_ids)
                    token_type_ids = [0] * max_seq_len
                    padding = [self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]] * (
                                max_seq_len - len(input_ids))
                    input_ids += padding
                    input_mask += [0] * len(padding)

                    if ex_index % 10000 == 0:
                        if self.args.symmetric:
                            logger.info("Writing example %d of %d" % (ex_index + i, len(examples) * 2))
                        else:
                            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    if ex_index < 10:
                        logger.info("*** Example ***")
                        logger.info(f"id: {ex.docId}")
                        logger.info("subtokens: %s" % " ".join(
                            [x for x in tokens]))
                        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                        logger.info("label: %s" % label)
                        if self.args.siamese:
                            start_end_positions = [tuple([trgt_id] + positions)]
                        else:
                            start_end_positions = [tuple([0] + positions[:2]), tuple([1] + positions[2:])]
                        for (trgt_id, start, end) in start_end_positions:
                            if start and end:
                                logger.info(f"Target{trgt_id}: {' '.join(tokens[start:end])}")

                    features.append(
                        WiCFeature2(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            syn_label=syn_label_to_id[label],
                            positions=positions,
                            example=ex
                            )
                        )
        logger.info("Not fitted examples percentage: %s" % str(num_too_long_exs / len(features) * 100.0))
        logger.info("Skipped examples percentage: %s" % str(skipped / len(features) * 100.0))
        return features


    def predict(
            self, eval_dataloader, output_dir,
            eval_features,
            compute_metrics=True
        ):
        self.eval()
        device = torch.device('cuda') if self.args.use_cuda else torch.device('cpu')

        metrics = defaultdict(float)
        nb_eval_steps = 0
        syns_preds = []

        for batch_id, batch in enumerate(tqdm(
            eval_dataloader, total=len(eval_dataloader),
            desc='validation ... '
        )):

            batch = tuple([elem.to(device) for elem in batch])

            input_ids, input_mask, token_type_ids, b_syn_labels, b_positions = batch

            with torch.no_grad():
                loss, syn_logits = self(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=input_mask,
                    input_labels={
                        'syn_labels': b_syn_labels,
                        'positions': b_positions
                        }
                    )

            if compute_metrics:
                for key, value in loss.items():
                    metrics[f'{key}_loss'] += value.mean().item()

            nb_eval_steps += 1
            syns_preds.append(syn_logits.detach().cpu().numpy())

        syns_scores = np.concatenate(syns_preds, axis=0)  # n_examples x 2 or n_examples x 1
        if syns_scores.shape[-1] != 1:
            syns_preds = np.argmax(syns_scores, axis=1)  # n_examples
        else:
            syns_preds = np.zeros(syns_scores.shape, dtype=int)
            syns_preds[syns_scores > 0.5] = 1

        predictions = defaultdict(lambda: defaultdict(list))
        golds = defaultdict(lambda: defaultdict(list))
        scores = defaultdict(lambda: defaultdict(list))

        gold_wsds = defaultdict(set)
        pred_wsds = defaultdict(lambda: (-1, '?'))

        syn_ids_to_label = {0: 'F', 1: 'T'}
        if self.args.siamese:
            features = [feature for i, feature in enumerate(eval_features) if i % 2]
        else:
            features = eval_features
        for ex_id, (ex_feature, ex_syn_preds, ex_scores) in enumerate(zip(features, syns_preds, syns_scores)):
            example = ex_feature.example
            docId = example.docId
            posInDoc = int(docId.split('.')[-1])
            docId = '.'.join(docId.split('.')[:-1])
            syn_pred = syn_ids_to_label[ex_syn_preds.item()]
            predictions[docId][posInDoc].append(syn_pred)
            golds[docId][posInDoc].append(example.label)
            # scores for positive class
            if len(ex_scores) > 1:
                scores[docId][posInDoc].append(softmax(ex_scores)[-1])
            else:
                scores[docId][posInDoc].append(ex_scores[0])

            if 'semcor' in docId:
                if example.label == 'T':
                    gold_wsds[example.target_id].add(example.sense_id)
                sense_score, sense_id = pred_wsds[example.target_id]
                if max(scores[docId][posInDoc]) > sense_score:
                    pred_wsds[example.target_id] = (max(scores[docId][posInDoc]), example.sense_id)


        if os.path.exists(output_dir):
            os.system(f'rm -r {output_dir}/*')
        else:
            os.makedirs(output_dir, exist_ok=True)

        for docId, doc_preds in predictions.items():
            doc_scores = scores[docId]
            print(f'saving predictions for part: {docId}')
            prediction = [{'id': f'{docId}.{pos}', 'tag': 'F' if 'F' in doc_preds[pos] else 'T'} for pos in
                          sorted(doc_preds)]
            prediction_file = os.path.join(output_dir, docId)
            json.dump(prediction, open(prediction_file, 'w'))
            prediction = [{'id': f'{docId}.{pos}', 'score': [str(x) for x in doc_scores[pos]]} for pos in
                          sorted(doc_preds)]
            prediction_file = os.path.join(output_dir, f'{docId}.scores')
            json.dump(prediction, open(prediction_file, 'w'))

        compute_nen_scores = False
        if compute_metrics:
            for key in metrics:
                metrics[key] /= nb_eval_steps
            mean_non_english = []
            for docId, doc_preds in predictions.items():
                doc_golds = golds[docId]
                keys = list(doc_golds.keys())
                doc_golds = [doc_golds[key][0] for key in keys]
                doc_preds = ['F' if 'F' in doc_preds[key] else 'T' for key in keys]
                metrics[f'accuracy.{docId}.score'] = accuracy_score(doc_golds, doc_preds)
                if 'en-en' not in docId:
                    mean_non_english.append(metrics[f'accuracy.{docId}.score'])
                else:
                    compute_nen_scores = True
                if 'semcor' in docId:
                    pred_wsds = {k: {v[1]} for k, v in pred_wsds.items()}
                    p, r, f = wsd_framework_score(gold_wsds, pred_wsds)
                    # for k in gold_wsds:
                    #     print(k, gold_wsds[k], pred_wsds[k])
                    # print(p, r, f)
                    metrics[f'f1_score.{docId}.score'] = f
                    metrics[f'p_score.{docId}.score'] = p
                    metrics[f'r_score.{docId}.score'] = r
            if mean_non_english and compute_nen_scores:
                metrics[f'accuracy.nen-nen.score'] = sum(mean_non_english) / len(mean_non_english)
        else:
            metrics = {}

        self.train()

        return metrics
