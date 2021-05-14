"""
Training script
"""

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torchtext.data.metrics import bleu_score
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
    idx2word,
)

from arguments import ArgumentParser
from bertadam import BertAdam
from datasets import ShapeWorld, extract_features
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from lxmert import Lxmert
from models import ImageRep, TextRep, TextProposal, ExWrapper
from models import MultimodalRep,MultimodalDeepRep
from models import DotPScorer, BilinearScorer
from vision import Conv4NP, ResNet18
from tre import AddComp, MulComp, CosDist, L1Dist, L2Dist, tre
from retrievers import construct_dict, gen_retriever
import matplotlib.pyplot as plt

TRE_COMP_FNS = {
    'add': AddComp,
    'mul': MulComp,
}

TRE_ERR_FNS = {
    'cos': CosDist,
    'l1': L1Dist,
    'l2': L2Dist,
}


def combine_feats(all_feats):
    """
    Combine feats like language, mask them, and get vocab
    """
    vocab = {}
    max_feat_len = max(len(f) for f in all_feats)
    feats_t = torch.zeros(len(all_feats), max_feat_len, dtype=torch.int64)
    feats_mask = torch.zeros(len(all_feats), max_feat_len, dtype=torch.uint8)
    for feat_i, feat in enumerate(all_feats):
        for j, f in enumerate(feat):
            if f not in vocab:
                vocab[f] = len(vocab)
            i_f = vocab[f]
            feats_t[feat_i, j] = i_f
            feats_mask[feat_i, j] = 1
    return feats_t, feats_mask, vocab


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    # train dataset will return (image, label, hint_input, hint_target, hint_length)
    precomputed_features = True # args.backbone == 'vgg16_fixed' TODO
    preprocess = False # args.backbone == 'resnet18' or args.backbone == 'lxmert' TODO
    train_dataset = ShapeWorld(
        split='train',
        vocab=None,
        augment=True,
        precomputed_features=precomputed_features,
        max_size=args.max_train,
        preprocess=preprocess,
        noise=args.noise,
        class_noise_weight=args.class_noise_weight,
        fixed_noise_colors=args.fixed_noise_colors,
        fixed_noise_colors_max_rgb=args.fixed_noise_colors_max_rgb,
        noise_type=args.noise_type,
        data_dir=args.data_dir,
        language_filter=args.language_filter,
        shuffle_words=args.shuffle_words,
        shuffle_captions=args.shuffle_captions)
    train_vocab = train_dataset.vocab
    train_vocab_size = train_dataset.vocab_size
    train_max_length = train_dataset.max_length
    train_w2i, train_i2w = train_vocab['w2i'], train_vocab['i2w']
    pad_index = train_w2i[PAD_TOKEN]
    sos_index = train_w2i[SOS_TOKEN]
    eos_index = train_w2i[EOS_TOKEN]
    test_class_noise_weight = 0.0
    if args.noise_at_test:
        test_noise = args.noise
    else:
        test_noise = 0.0
    val_dataset = ShapeWorld(split='val',
                             precomputed_features=precomputed_features,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             noise=test_noise,
                             class_noise_weight=0.0,
                             noise_type=args.noise_type,
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=precomputed_features,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              noise=test_noise,
                              class_noise_weight=0.0,
                              noise_type=args.noise_type,
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        has_same = True
    except RuntimeError:
        has_same = False
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    if has_same:
        val_same_loader = torch.utils.data.DataLoader(
            val_same_dataset, batch_size=args.batch_size, shuffle=False)
        test_same_loader = torch.utils.data.DataLoader(
            test_same_dataset, batch_size=args.batch_size, shuffle=False)

    data_loader_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'val_same': val_same_loader if has_same else None,
        'test_same': test_same_loader if has_same else None,
    }

    if args.backbone == 'vgg16_fixed':
        backbone_model = None
    elif args.backbone == 'conv4':
        backbone_model = Conv4NP()
    elif args.backbone == 'resnet18':
        backbone_model = ResNet18()
    elif args.backbone == 'lxmert':
        backbone_model = Lxmert(30522, 516, 4608, 2, args.initializer_range, pretrained=False)
    else:
        raise NotImplementedError(args.backbone)

    if args.hint_retriever:
        image_model = ExWrapper(ImageRep(backbone_model, hidden_size=512), retrieve_mode=True)
    elif args.backbone == 'lxmert':
        image_model = backbone_model
    else:
        image_model = ExWrapper(ImageRep(backbone_model, hidden_size=512))
    image_model = image_model.to(device)
    params_to_optimize = list(image_model.parameters())

    if args.comparison == 'dotp':
        scorer_model = DotPScorer()
    elif args.comparison == 'bilinear':
        # FIXME: This won't work with --poe
        scorer_model = BilinearScorer(512,
                                      dropout=args.dropout,
                                      identity_debug=args.debug_bilinear)
    else:
        raise NotImplementedError
    scorer_model = scorer_model.to(device)
    params_to_optimize.extend(scorer_model.parameters())

    if args.use_hyp:
        embedding_model = nn.Embedding(train_vocab_size, 512)

    if args.decode_hyp:
        proposal_model = TextProposal(embedding_model, hidden_size=512)
        proposal_model = proposal_model.to(device)
        params_to_optimize.extend(proposal_model.parameters())

    if args.encode_hyp:
        if args.hint_retriever:
            hint_model = TextRep(embedding_model, hidden_size=512, retrieve_mode=True)
        else:
            hint_model = TextRep(embedding_model, hidden_size=512)
        hint_model = hint_model.to(device)
        params_to_optimize.extend(hint_model.parameters())

    if args.multimodal_concept:
        multimodal_model = MultimodalDeepRep()
        multimodal_model = multimodal_model.to(device)
        params_to_optimize.extend(multimodal_model.parameters())

    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD,
        'bertadam': BertAdam
    }[args.optimizer]

    t_total = int(100 * args.epochs)
    optimizer = optfunc(params_to_optimize, lr=args.lr, warmup=args.warmup_ratio, t_total=t_total)
    
    # initialize weight and bias
    #wandb.init(project='lsl', entity='bhy070418s')
    wandb.init(project='easton_dev', entity='lsl')
    config = wandb.config
    config.learning_rate = args.lr

    wandb.watch(image_model)

    cross_entropy = nn.CrossEntropyLoss()
    def train(epoch, n_steps=100):
        image_model.train()
        scorer_model.train()
        if args.decode_hyp:
            proposal_model.train()
        if args.encode_hyp:
            hint_model.train()
        if args.multimodal_concept:
            multimodal_model.train()

        loss_total = 0
        pbar = tqdm(total=n_steps)
        for batch_idx in range(n_steps):
            examples, image, label, hint_tokens, attention_masks = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device)
            image = image.to(device)
            label = label.to(device)
            batch_size = len(image)
            n_ex = examples.shape[1]

            # Learn representations of images and examples
            hint_tokens = hint_tokens.to(device)
            attention_masks = attention_masks.to(device)
            
            concat_images = torch.cat((examples, torch.unsqueeze(image, dim=1)), dim=1)
            score = image_model(concat_images, input_ids=hint_tokens, attention_mask=attention_masks)
            # Use concept to compute prediction loss
            # (how well does example repr match image repr)?

            loss = cross_entropy(score, label)
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f}'.format(
                    epoch, loss.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tLoss: {:.4f}'.format(
            '(train)', epoch, loss_total))

        return loss_total

    def test(epoch, split='train', hint_rep_dict=None):
        image_model.eval()
        scorer_model.eval()
        if args.infer_hyp:
            # If predicting hyp only, ignore encode/decode models for eval
            proposal_model.eval()
            hint_model.eval()
            if args.multimodal_concept:
                multimodal_model.eval()

        accuracy_meter = AverageMeter(raw=True)
        precision_meter = AverageMeter(raw=True)
        recall_meter = AverageMeter(raw=True)
        retrival_acc_meter = AverageMeter(raw=True)
        bleu_meter_n1 = AverageMeter(raw=True)
        bleu_meter_n2 = AverageMeter(raw=True)
        bleu_meter_n3 = AverageMeter(raw=True)
        bleu_meter_n4 = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_tokens, attention_masks  in data_loader:
                
                examples = examples.to(device)
                image = image.to(device)
                label = label.numpy()
                label_np = label.astype(np.uint8)
                batch_size = len(image)


                hint_tokens = hint_tokens.to(device)
                attention_masks = attention_masks.to(device)

                concat_images = torch.cat((examples, torch.unsqueeze(image, dim=1)), dim=1)
                score = image_model(concat_images, input_ids=hint_tokens, attention_mask=attention_masks)
                # Compare image directly to example rep
                label_hat = torch.argmax(score, dim=1)
                label_hat = label_hat.cpu().numpy()
                accuracy = accuracy_score(label_np, label_hat)
                precision = precision_score(label_np, label_hat, zero_division=0)
                recall = recall_score(label_np, label_hat, zero_division=0)
                accuracy_meter.update(accuracy,
                                      batch_size,
                                      raw_scores=(label_hat == label_np))
                precision_meter.update(precision,
                                      batch_size,
                                      raw_scores=[precision])
                recall_meter.update(recall,
                                      batch_size,
                                      raw_scores=[recall])

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}\tPrecision: {:.4f}\tRecall: {:.4f}\
            \tBLEU_n1 Score: {:.4f}\tBLEU_n2 Score: {:.4f} \tBLEU_n3 Score: {:.4f}\tBLEU_n4 Score: {:.4f}\tRetrieval Accuracy: {:.4f}'.format(
            '({})'.format(split), epoch, accuracy_meter.avg, precision_meter.avg, recall_meter.avg, \
                bleu_meter_n1.avg, bleu_meter_n2.avg, bleu_meter_n3.avg, bleu_meter_n4.avg, retrival_acc_meter.avg))
        return accuracy_meter.avg, accuracy_meter.raw_scores, precision_meter.avg, recall_meter.avg, \
            bleu_meter_n1.avg, bleu_meter_n2.avg, bleu_meter_n3.avg, bleu_meter_n4.avg

    tre_comp_fn = TRE_COMP_FNS[args.tre_comp]()
    tre_err_fn = TRE_ERR_FNS[args.tre_err]()


    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_val_tre = 0
    best_val_tre_std = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    lowest_val_tre = 1e10
    lowest_val_tre_std = 0
    metrics = defaultdict(lambda: [])

    val_acc_collection = []
    bleu_n1_collection = []
    bleu_n2_collection = []
    bleu_n3_collection = []
    bleu_n4_collection = []

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    hint_rep_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        if epoch % 10 != 1 :
            continue
        # storing seen concepts' hint representations
        if args.hint_retriever:
            train_dataset.augment = False # this is not gonna work if there are multiple workers
            hint_rep_dict = construct_dict(train_loader, image_model, hint_model)
            train_dataset.augment = True
        train_acc, _, train_prec, train_reca, *_ = test(epoch, 'train', hint_rep_dict)
        val_acc, _, val_prec, val_reca, *_ = test(epoch, 'val', hint_rep_dict)
      
        val_tre, val_tre_std = 0.0, 0.0

        test_acc, test_raw_scores, test_prec, test_reca, \
            test_bleu_n1, test_bleu_n2, test_bleu_n3, test_bleu_n4 = test(epoch, 'test', hint_rep_dict)
        if has_same:
            val_same_acc, _, val_same_prec, val_same_reca, *_ = test(epoch, 'val_same', hint_rep_dict)
            test_same_acc, test_same_raw_scores, test_same_prec, test_same_reca,\
                test_same_bleu_n1, test_same_bleu_n2, test_same_bleu_n3, test_same_bleu_n4 = test(epoch, 'test_same', hint_rep_dict)    
            all_test_raw_scores = test_raw_scores + test_same_raw_scores
        else:
            val_same_acc = val_acc
            test_same_acc = test_acc
            all_test_raw_scores = test_raw_scores

        wandb.log({"loss": train_loss, 'train_acc': train_acc, 'train_prec': train_prec, 'train_reca': train_reca,\
            'val_same_acc': val_same_acc, 'val_same_prec': val_same_prec, 'val_same_reca': val_same_reca,\
            'val_acc': val_acc,'val_prec': val_prec, 'val_reca': val_reca,\
            'test_same_acc': test_same_acc, 'test_same_prec': test_same_prec, 'test_same_reca': test_same_reca,\
            'test_acc': test_acc, 'test_prec': test_prec, 'test_reca': test_reca})
        
        # Compute confidence intervals
        n_test = len(all_test_raw_scores)
        test_acc_ci = 1.96 * np.std(all_test_raw_scores) / np.sqrt(n_test)

        epoch_acc = (val_acc + val_same_acc) / 2
        is_best_epoch = epoch_acc > (best_val_acc + best_val_same_acc) / 2
        average_bleu_n1 = (test_same_bleu_n1 + test_bleu_n1) / 2
        average_bleu_n2 = (test_same_bleu_n2 + test_bleu_n2) / 2
        average_bleu_n3 = (test_same_bleu_n3 + test_bleu_n3) / 2
        average_bleu_n4 = (test_same_bleu_n4 + test_bleu_n4) / 2
        val_acc_collection.append(epoch_acc)
        bleu_n1_collection.append(average_bleu_n1)
        bleu_n4_collection.append(average_bleu_n4)
        bleu_n2_collection.append(average_bleu_n2)
        bleu_n3_collection.append(average_bleu_n3)

        if is_best_epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': image_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "./best_model.pt")
            best_epoch = epoch
            best_epoch_acc = epoch_acc
            best_val_acc = val_acc
            best_val_same_acc = val_same_acc
            best_val_tre = val_tre
            best_val_tre_std = val_tre_std
            best_test_acc = test_acc
            best_test_same_acc = test_same_acc
            best_test_acc_ci = test_acc_ci
            best_test_bleu_n1 = average_bleu_n1
            best_test_bleu_n2 = average_bleu_n2
            best_test_bleu_n3 = average_bleu_n3
            best_test_bleu_n4 = average_bleu_n4
        if val_tre < lowest_val_tre:
            lowest_val_tre = val_tre
            lowest_val_tre_std = val_tre_std

        if args.save_checkpoint:
            raise NotImplementedError

        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_same_acc'].append(val_same_acc)
        metrics['val_tre'].append(val_tre)
        metrics['val_tre_std'].append(val_tre_std)
        metrics['test_acc'].append(test_acc)
        metrics['test_same_acc'].append(test_same_acc)
        metrics['test_acc_ci'].append(test_acc_ci)

        metrics = dict(metrics)
        # Assign best accs
        metrics['best_epoch'] = best_epoch
        metrics['best_val_acc'] = best_val_acc
        metrics['best_val_same_acc'] = best_val_same_acc
        metrics['best_val_tre'] = best_val_tre
        metrics['best_val_tre_std'] = best_val_tre_std
        metrics['best_test_acc'] = best_test_acc
        metrics['best_test_same_acc'] = best_test_same_acc
        metrics['best_test_acc_ci'] = best_test_acc_ci
        metrics['lowest_val_tre'] = lowest_val_tre
        metrics['lowest_val_tre_std'] = lowest_val_tre_std
        metrics['has_same'] = has_same
        save_defaultdict_to_fs(metrics,
                               os.path.join(args.exp_dir, 'metrics.json'))
    torch.save({
            'epoch': epoch,
            'model_state_dict': image_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "./last_epoch.pt")
    print('====> DONE')
    print('====> BEST EPOCH: {}'.format(best_epoch))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val)', best_epoch, best_val_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_same)', best_epoch, best_val_same_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}\tCI: {:.4f}'.format(
        '(best_test)', best_epoch, best_test_acc, best_test_acc_ci))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_same)', best_epoch, best_test_same_acc))
    print('====>')
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_avg)', best_epoch, (best_val_acc + best_val_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_avg)', best_epoch,
        (best_test_acc + best_test_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy CI: {:.4f}'.format(
        '(best_test_acc_ci)', best_epoch,
        best_test_acc_ci))
    print('====> {:>17}\tEpoch: {}\tBLEU_N1: {:.4f}'.format(
        '(best_test_bleu_n1)', best_epoch,
        best_test_bleu_n1))
    print('====> {:>17}\tEpoch: {}\tBLEU_N1: {:.4f}'.format(
        '(best_test_bleu_n2)', best_epoch,
        best_test_bleu_n2))
    print('====> {:>17}\tEpoch: {}\tBLEU_N4: {:.4f}'.format(
        '(best_test_bleu_n3)', best_epoch,
        best_test_bleu_n3))
    print('====> {:>17}\tEpoch: {}\tBLEU_N4: {:.4f}'.format(
        '(best_test_bleu_n4)', best_epoch,
        best_test_bleu_n4))
    if args.plot_bleu_score:
        x = (np.array(range(len(val_acc_collection))) + 1)
        plt.plot(x, val_acc_collection, label = "validation accuracy")
        plt.plot(x, bleu_n1_collection, label = "bleu n=1")
        plt.plot(x, bleu_n2_collection, label = "bleu n=2")
        plt.plot(x, bleu_n3_collection, label = "bleu n=3")
        plt.plot(x, bleu_n4_collection, label = "bleu n=4")
        plt.xlabel('epoch')
        plt.ylabel('%')
        plt.legend( loc="upper right")
        plt.savefig('accuracy_vs_bleu_original.png')

