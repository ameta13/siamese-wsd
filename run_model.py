import argparse
import logging
import os
import random
import time
import json
from datetime import datetime
import tempfile
import shutil

from glob import glob
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.nn import CosineSimilarity
from scipy.stats import spearmanr

from torch.nn import CrossEntropyLoss

from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
)
from transformers.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME, CONFIG_NAME
)

from tqdm import tqdm
from models.utils import (
    configs, DataProcessor, models
)
from models.utils import get_dataloader_and_tensors
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report, accuracy_score
)
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def main(args):
    if not args.do_eval:
        assert all([x in ['true', 'false'] for x in [args.use_cuda, args.symmetric, args.linear_head, args.siamese]])
        args.use_cuda = args.use_cuda.lower() == 'true'
        args.symmetric = args.symmetric.lower() == 'true'
        args.linear_head = args.linear_head.lower() == 'true'
        args.siamese = args.siamese.lower() == 'true'

    if args.siamese:
        assert args.train_batch_size % 2 == 0, 'train batch size should be even in siamese mode'
        assert not args.symmetric

    if args.do_train and os.path.exists(args.output_dir):
        model_weights = glob(os.path.join(args.output_dir, '*.bin'))
        if model_weights:
            print(f'{model_weights}: already computed: skipping ...')
            return
        else:
            print(f'already existing {args.output_dir}. but without model weights ...')
            return

    device = torch.device("cuda" if args.use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps parameter should be >= 1"
        )

    if args.do_train:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True."
        )

    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'nen-nen-weights'))
    elif args.do_train or args.do_validation:
        raise ValueError(f'{args.output_dir} already exists')

    suffix = datetime.now().isoformat().replace('-', '_').replace(
        ':', '_').split('.')[0].replace('T', '-')

    if args.do_train:
        train_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'train'
            )
        )
        dev_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'dev'
            )
        )
        test_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'test'
            )
        )

        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"train_logs_{suffix}.log"), 'w')
        )
    else:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.ckpt_path, f"eval_logs_{suffix}.log"), 'w')
        )

    logger.info(json.dumps(vars(args), indent=4))
    if args.do_train:
        json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), 'w'), indent=4)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    model_name = args.model_name
    data_processor = DataProcessor()

    train_dir = os.path.join(args.data_dir, 'train/')
    dev_dir = os.path.join(args.data_dir, 'dev')

    if args.do_train:
        config = configs[args.model_name]
        config = config.from_pretrained(
            args.model_name,
            hidden_dropout_prob=args.dropout
        )
        if args.ckpt_path != '':
            model_path = args.ckpt_path
        else:
            model_path = args.model_name

        model = models[model_name]
        model = model.from_pretrained(
            model_path, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            args=args,
            data_processor=data_processor,
            config=config
        )

        if args.freeze_featurizer:
            trainable_weights = []
            for name, parameter in model.named_parameters():
                # if name not in ['syn_clf.bn1.weight', 'syn_clf.bn1.bias', 'syn_clf.bn1.running_mean', 'syn_clf.bn1.running_var', 'syn_clf.dense.weight', 'syn_clf.dense.bias', 'syn_clf.out_proj.weight', 'syn_clf.out_proj.bias']:
                if name.startswith('roberta'):
                    parameter.requires_grad = False
                else:
                    trainable_weights.append(name)
            logger.info(f'trainable weights: {trainable_weights}')

        model.to(device)

        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)
                ],
                'weight_decay': float(args.weight_decay)
            },
            {
                'params': [
                    param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(args.learning_rate),
            eps=1e-6,
            betas=(0.9, 0.98),
            correct_bias=True
        )

        train_features = model.convert_dataset_to_features(
            train_dir, logger
        )

        train_dataloader = \
            get_dataloader_and_tensors(train_features, args.train_batch_size, 'siamese_random' if args.siamese else 'random')
        train_batches_len = len(train_dataloader)

        num_train_optimization_steps = \
            train_batches_len // args.gradient_accumulation_steps * \
                args.num_train_epochs

        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        if args.lr_scheduler == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps
            )
        elif args.lr_scheduler == 'constant_warmup':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps
            )

        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=args.fp16_opt_level,
                # loss_scale=args.loss_scale,
                # min_loss_scale=args.fp16_min_loss_scale,
                # max_loss_scale=args.fp16_max_loss_scale,
            )
        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_features = None

        if args.do_validation:
            dev_features = model.convert_dataset_to_features(
                dev_dir, logger
            )
            logger.info("***** Dev *****")
            logger.info("  Num examples = %d", len(dev_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            dev_dataloader = \
                get_dataloader_and_tensors(dev_features, args.eval_batch_size, 'sequential')
            test_dir = os.path.join(args.data_dir, 'test/')
            if os.path.exists(test_dir):
                test_features = model.convert_dataset_to_features(
                    test_dir, logger
                )
                logger.info("***** Test *****")
                logger.info("  Num examples = %d", len(test_features))
                logger.info("  Batch size = %d", args.eval_batch_size)

                test_dataloader = \
                    get_dataloader_and_tensors(test_features, args.eval_batch_size, 'sequential')

        best_result = defaultdict(float)

        eval_step = max(1, train_batches_len // args.eval_per_epoch)

        start_time = time.time()
        global_step = 0

        lr = float(args.learning_rate)
        for epoch in range(1, 1 + args.num_train_epochs):
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            cur_train_loss = defaultdict(float)

            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, scheduler.get_lr()[0]))

            train_bar = tqdm(
                train_dataloader, total=train_batches_len,
                desc='training ... '
            )
            for step, batch in enumerate(
                train_bar
            ):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, \
                    syn_labels, positions = batch

                train_loss, _ = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=input_mask,
                    input_labels={'syn_labels': syn_labels, 'positions': positions}
                )
                loss = train_loss['total'].mean().item()
                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()

                train_bar.set_description(f'training... [epoch == {epoch} / {args.num_train_epochs}, loss == {loss}]')

                loss_to_optimize = train_loss['total']

                if args.gradient_accumulation_steps > 1:
                    loss_to_optimize = \
                        loss_to_optimize / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss_to_optimize, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_to_optimize.backward()

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer),
                            args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    # optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if args.do_validation and (step + 1) % eval_step == 0:
                    logger.info(
                        'Ep: {}, Stp: {}/{}, usd_t={:.2f}s, loss={:.6f}'.format(
                            epoch, step + 1, train_batches_len,
                            time.time() - start_time, tr_loss / nb_tr_steps
                        )
                    )
                    cur_train_mean_loss = {}
                    for key, value in cur_train_loss.items():
                        cur_train_mean_loss[f'{key}_loss'] = \
                            value / nb_tr_steps

                    dev_predictions = os.path.join(args.output_dir, 'dev_predictions')

                    metrics = model.predict(
                        dev_dataloader, dev_predictions,
                        dev_features, compute_metrics=True
                    )

                    metrics['global_step'] = global_step
                    metrics['epoch'] = epoch
                    metrics['learning_rate'] = scheduler.get_lr()[0]
                    metrics['batch_size'] = \
                        args.train_batch_size * args.gradient_accumulation_steps

                    for key, value in metrics.items():
                        dev_writer.add_scalar(key, value, global_step)
                    for key, value in cur_train_mean_loss.items():
                        train_writer.add_scalar(key, value, global_step)
                    scores_to_logger = tuple([round(metrics[save_by_score] * 100.0, 2) for save_by_score in args.save_by_score.split('+')])
                    logger.info(f"dev %s (lr=%s, epoch=%d): %s" %
                        (
                            args.save_by_score,
                            str(scheduler.get_lr()[0]), epoch,
                            scores_to_logger
                        )
                    )

                    improved_parts = [part  for part in metrics if part.endswith('.score') and metrics[part] > args.start_save_threshold and metrics[part] > best_result[part]]
                    if improved_parts:
                        best_dev_predictions = os.path.join(args.output_dir, 'best-dev-predictions')
                        dev_predictions = os.path.join(args.output_dir, 'dev_predictions')
                        os.makedirs(best_dev_predictions, exist_ok=True)
                        os.makedirs(dev_predictions, exist_ok=True)
                        for part in improved_parts:
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f -> %.2f" %
                                (
                                    part,
                                    str(scheduler.get_lr()[0]), epoch,
                                    best_result[part] * 100.0, metrics[part] * 100.0
                                )
                            )
                            best_result[part] = metrics[part]
                            dev_writer.add_scalar('best_' + part, metrics[part], global_step)

                            if [save_weight for save_weight in args.save_by_score.split('+') if save_weight == part]:
                                os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)
                                output_model_file = os.path.join(
                                    args.output_dir, part,
                                    WEIGHTS_NAME
                                )
                                save_model(args, model, output_model_file, metrics)
                            best_dev_files = [file.split('/')[-1] for file in glob(f'{dev_predictions}/*') if part.split('.')[1] in file]
                            for dev_file in best_dev_files:
                                logger.info(f'{dev_predictions}/{dev_file} -> {best_dev_predictions}/')
                                os.system(f'cp {dev_predictions}/{dev_file} {best_dev_predictions}/')

                        if args.log_test_metrics and os.path.exists(test_dir):
                            test_predictions = os.path.join(args.output_dir, 'test_predictions')
                            test_metrics = model.predict(
                                test_dataloader, test_predictions,
                                test_features, compute_metrics=True
                            )
                            best_test_predictions = os.path.join(args.output_dir, 'best-test-predictions')
                            os.makedirs(best_test_predictions, exist_ok=True)
                            corresp_test_files = [file.split('/')[-1] for file in glob(f'{test_predictions}/*') if any([part.split('.')[1] in file for part in improved_parts])]
                            for test_file in corresp_test_files:
                                logger.info(f'{test_predictions}/{test_file} -> {best_test_predictions}/')
                                os.system(f'cp {test_predictions}/{test_file} {best_test_predictions}/')

                            for key, value in test_metrics.items():
                                if key.endswith('.score'):
                                    test_writer.add_scalar(key, value, global_step)
                                if key in improved_parts:
                                    test_writer.add_scalar('best_' + key, value, global_step)
                        if any(['nen-nen.score' in part for part in improved_parts]):
                            best_dev_nen_nen_path = os.path.join(args.output_dir, 'best-dev-nen-nen-predictions')
                            os.makedirs(best_dev_nen_nen_path, exist_ok=True)
                            os.system(f'mv {dev_predictions}/* {best_dev_nen_nen_path}/')
                            if args.log_test_metrics and os.path.exists(test_dir):
                                best_test_nen_nen_path = os.path.join(args.output_dir, 'best-test-nen-nen-predictions')
                                os.makedirs(best_test_nen_nen_path, exist_ok=True)
                                os.system(f'mv {test_predictions}/* {best_test_nen_nen_path}/')


    if args.do_eval:
        assert args.ckpt_path != '', 'in do_eval mode ckpt_path should be specified'
        test_dir = args.eval_input_dir
        config = configs[model_name].from_pretrained(model_name)
        model = models[model_name]
        model = model.from_pretrained(args.ckpt_path, args=args, data_processor=data_processor, config=config)
        model.to(device)
        test_features = model.convert_dataset_to_features(
            test_dir, logger
        )
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_dataloader = \
            get_dataloader_and_tensors(test_features, args.eval_batch_size, 'sequential')

        metrics = model.predict(
            test_dataloader,
            os.path.join(args.output_dir, args.eval_output_dir),
            test_features,
            compute_metrics=True
        )
        logger.info(json.dumps(metrics, indent=4))
        with open(os.path.join(args.output_dir, args.eval_output_dir,'metrics.txt'), 'w') as outp:
            json.dump(metrics, outp, indent=4)


def save_model(args, model, output_model_file, metrics):
    start = time.time()
    model_to_save = \
        model.module if hasattr(model, 'module') else model

    output_config_file = os.path.join(
        args.output_dir, CONFIG_NAME
    )
    torch.save(
        model_to_save.state_dict(), output_model_file
    )
    model_to_save.config.to_json_file(
        output_config_file
    )
    with open(output_model_file+'.txt', 'w') as outp:
        print(metrics, file=outp)
    print(f'model saved in {time.time() - start} seconds to {output_model_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='data/', type=str,
                        help="The directory where train and dev directories are located.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_per_epoch", default=4, type=int,
                        help="How many times to do validation on dev set per epoch")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup.\n"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="train batch size")
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int,
                        help="gradinent accumulation steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="maximal gradient norm")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate")
    parser.add_argument("--start_save_threshold", default=0.0, type=float,
                        help="accuracy threshold to start save models")
    parser.add_argument("--loss", type=str, default='crossentropy_loss',
                        choices=['crossentropy_loss', 'mse_loss', 'cosine_similarity', 'mseplus_loss'])
    parser.add_argument("--lr_scheduler", type=str, default='linear_warmup',
                        choices=['constant_warmup', 'linear_warmup'])
    parser.add_argument("--model_name", type=str, default='xlm-roberta-large',
                        choices=['xlm-roberta-large', 'xlm-roberta-base'])

    parser.add_argument("--pool_type", type=str, default='mean')

    parser.add_argument("--save_by_score", type=str, default='accuracy.en-en.score')
    parser.add_argument("--ckpt_path", type=str, default='', help='Path to directory containig pytorch.bin checkpoint')
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--target_embeddings", type=str, default='comb_dmn')
    parser.add_argument("--head_batchnorm", type=int, default=0)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--linear_head", type=str, default='false')

    parser.add_argument("--do_train", action='store_true', help='Whether to run training')
    parser.add_argument("--do_validation", action='store_true',
                        help='Whether to validate model during training process')
    parser.add_argument("--do_eval", action='store_true', help='Whether to run evaluation')
    parser.add_argument("--use_cuda", default='true', type=str, help='Whether to use GPU')
    parser.add_argument("--symmetric", default='true', type=str, help='Whether to augment data by symmetry')
    parser.add_argument("--mask_targets", action='store_true',
                        help='Whether to replace target words in context by mask tokens')
    parser.add_argument("--train_scd", action='store_true', help='Whether to train semantic change detection model')
    parser.add_argument("--eval_input_dir", default='data/wic/test', help='Directory containing .data files to predict')
    parser.add_argument("--eval_output_dir", default='best_eval_test_predictions', help='Directory name where predictions will be saved')
    parser.add_argument("--siamese", type=str, default='false')

    parser.add_argument("--fp16", action='store_true', help='train in mixed precision mode')
    parser.add_argument("--fp16-opt-level", default="O3", choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument("--fp16-min-loss-scale", default=None)
    parser.add_argument("--loss_scale", default=None)
    parser.add_argument("--fp16-max-loss-scale", default=2.**24)
    parser.add_argument("--freeze_featurizer", action='store_true', help='train only classifier and freeze vectorizer weights')
    parser.add_argument("--log_test_metrics", action='store_true', help='Whether to run evaluation during validation time'
                                                                        ' on test set and write its metrics to tensorboard')

    parser.add_argument("--layers", type=str, default='-1', help='indexes of layers for vectorizer model'
                                                                 'examples: -1,-2,-3,-4 for last for layers')

    parsed_args = parser.parse_args()
    if parsed_args.do_eval:
        args_path = os.path.join(parsed_args.ckpt_path, 'args.json')
        if not os.path.exists(args_path):
            args_path = os.path.join('/'.join((parsed_args.ckpt_path.rstrip('/').split('/')[:-1])), 'args.json')
        new_args = json.load(open(args_path))
        for key, value in new_args.items():
            if key.startswith('do') or key in ['ckpt_path', 'eval_input_dir', 'eval_output_dir', 'output_dir', 'eval_batch_size']:
                continue
            setattr(parsed_args, key, value)
    main(parsed_args)
