'''
train
'''
import glob
import os
import random
import re
from functools import partial
from math import ceil

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from . import dataloader, model, util
from .model import decode_greedy

from code.timer import Timer
train_timer = Timer.get("train")

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class Arch:
    soft = 'soft'  # soft attention without input-feeding
    hard = 'hard'  # hard attention with dynamic programming without input-feeding
    hmm = 'hmm'  # 0th-order hard attention without input-feeding
    hmmfull = 'hmmfull'  # 1st-order hard attention without input-feeding


DEV = 'dev'
TEST = 'test'

model_classfactory = {
    (Arch.soft, False): model.TagTransducer,
    (Arch.hard, False): model.TagHardAttnTransducer,
    (Arch.hmm, True): model.MonoTagHMMTransducer,
    (Arch.hmmfull, False): model.TagFullHMMTransducer,
    (Arch.hmmfull, True): model.MonoTagFullHMMTransducer
}


class Trainer(object):
    '''docstring for Trainer.'''

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        print("device", self.device)
        self.model = None
        self.optimizer = None
        self.min_lr = 0
        self.scheduler = None
        self.evaluator = None
        self.last_devloss = float('inf')
        self.models = list()

    def load_data(self, source_lang, target_lang, src_vocab, trg_vocab, test=None, shuffle=False):
        assert self.data is None
        logger = self.logger
        # yapf: disable
        self.data = dataloader.TagSIGMORPHON2019Task1(source_lang=source_lang, target_lang=target_lang,
                                                      test=test, shuffle=shuffle, src_vocab=src_vocab,
                                                      trg_vocab=trg_vocab)
        # yapf: enable
        logger.info('src vocab size %d', self.data.source_vocab_size)
        logger.info('trg vocab size %d', self.data.target_vocab_size)
        logger.info('src vocab %r', self.data.source[:500])
        logger.info('trg vocab %r', self.data.target[:500])

    def build_model(self, embed_dim, dropout, src_hs, trg_hs, src_layer, trg_layer, wid_siz, arch, mono):
        assert self.model is None
        params = dict()
        params['src_vocab_size'] = self.data.source_vocab_size
        params['trg_vocab_size'] = self.data.target_vocab_size
        params['embed_dim'] = embed_dim
        params['dropout_p'] = dropout
        params['src_hid_size'] = src_hs
        params['trg_hid_size'] = trg_hs
        params['src_nb_layers'] = src_layer
        params['trg_nb_layers'] = trg_layer
        params['nb_attr'] = self.data.nb_attr
        params['wid_siz'] = wid_siz
        params['src_c2i'] = self.data.source_c2i
        params['trg_c2i'] = self.data.target_c2i
        params['attr_c2i'] = self.data.attr_c2i
        model_class = model_classfactory[(arch, mono)]
        self.model = model_class(**params)
        self.logger.info('number of attribute %d', self.model.nb_attr)
        self.logger.info('dec 1st rnn %r', self.model.dec_rnn.layers[0])
        self.logger.info('number of parameter %d',
                         self.model.count_nb_params())
        self.model = self.model.to(self.device)

    def load_model(self, model):
        assert self.model is None
        self.logger.info('load model in %s', model)
        self.model = torch.load(open(model, mode='rb'), map_location=self.device)
        self.model = self.model.to(self.device)
        epoch = int(model.split('_')[-1])
        return epoch

    def smart_load_model(self, model_prefix):
        assert self.model is None
        models = []
        for model in glob.glob(f'{model_prefix}.nll*'):
            res = re.findall(r'\w*_\d+\.?\d*', model)
            loss_, evals_, epoch_ = res[0].split('_'), res[1:-1], res[-1].split('_')
            assert loss_[0] == 'nll' and epoch_[0] == 'epoch'
            loss, epoch = float(loss_[1]), int(epoch_[1])
            evals = []
            for ev in evals_:
                ev = ev.split('_')
                evals.append(util.Eval(ev[0], ev[0], float(ev[1])))
            models.append((epoch, (model, loss, evals)))
        self.models = [x[1] for x in sorted(models)]
        return self.load_model(self.models[-1][0])

    def setup_training(self, optimizer, lr, momentum):
        assert self.model is not None
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr, momentum=momentum)
        elif optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        elif optimizer == 'amsgrad':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr, amsgrad=True)
        else:
            raise ValueError

    def setup_scheduler(self, min_lr, patience, cooldown, discount_factor):
        self.min_lr = min_lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=patience,
            cooldown=cooldown,
            factor=discount_factor,
            min_lr=min_lr)

    def save_training(self, model_fp):
        save_objs = (self.optimizer.state_dict(), self.scheduler.state_dict())
        torch.save(save_objs, open(f'{model_fp}.progress', 'wb'))

    def load_training(self, model_fp):
        assert self.model is not None
        optimizer_state, scheduler_state = torch.load(
            open(f'{model_fp}/model.progress', 'rb'))
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)

    def setup_evalutator(self):
        self.evaluator = util.BasicEvaluator()

    def train(self, epoch_idx, batch_size, max_norm, pre=False):
        logger, model, data = self.logger, self.model, self.data
        if pre:
            batch_sample = data.pretrain_batch_sample
            nb_batch = ceil(data.nb_pretrain / batch_size)
        else:
            batch_sample = data.train_batch_sample
            nb_batch = ceil(data.nb_train / batch_size)

        logger.info('At %d-th epoch with lr %f.', epoch_idx,
                    self.optimizer.param_groups[0]['lr'])
        model.train()
        print(f"{'pre' if pre else '' }train", epoch_idx)
        for src, src_mask, trg, _ in tqdm(batch_sample(batch_size), total=nb_batch):
            train_timer.task("out")
            # print(src, src_mask, trg)
            out = model(src, src_mask, trg)
            train_timer.task("loss")
            loss = model.loss(out, trg[1:])
            train_timer.task("zero grad")
            self.optimizer.zero_grad()
            train_timer.task("backward")
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            logger.debug('loss %f with total grad norm %f', loss,
                         util.grad_norm(model.parameters()))
            train_timer.task("opt")
            self.optimizer.step()
        train_timer.report()

    def iterate_batch(self, mode, batch_size):
        if mode == 'dev':
            return self.data.dev_batch_sample, ceil(
                self.data.nb_dev / batch_size)
        elif mode == 'test':
            return self.data.test_batch_sample, ceil(
                self.data.nb_test / batch_size)
        else:
            raise ValueError(f'wrong mode: {mode}')

    def calc_loss(self, mode, batch_size, epoch_idx=-1):
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        loss, cnt = 0, 0
        for src, src_mask, trg, _ in tqdm(sampler(batch_size), total=nb_batch):
            out = self.model(src, src_mask, trg)
            loss += self.model.loss(out, trg[1:]).item()
            cnt += 1
        loss = loss / cnt
        self.logger.info(
            'Average %s loss value per instance is %f at the end of epoch %d',
            mode, loss, epoch_idx)
        return loss

    def iterate_instance(self, mode):
        if mode == 'dev':
            return self.data.dev_sample, self.data.nb_dev
        elif mode == 'test':
            return self.data.test_sample, self.data.nb_test
        else:
            raise ValueError(f'wrong mode: {mode}')

    def evaluate(self, mode, epoch_idx=-1, decode_fn=decode_greedy):
        self.model.eval()
        sampler, nb_instance = self.iterate_instance(mode)
        results = self.evaluator.evaluate_all(sampler, nb_instance, self.model,
                                              decode_fn)
        for result in results:
            self.logger.info('%s %s is %f at the end of epoch %d', mode,
                             result.long_desc, result.res, epoch_idx)
        return results

    def decode(self, mode, write_fp, decode_fn=decode_greedy):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance(mode)
        with open(f'{write_fp}.{mode}.guess', 'w') as out_fp, \
             open(f'{write_fp}.{mode}.gold', 'w') as trg_fp:
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                out_fp.write(f'{"".join(pred)}\n')
                trg_fp.write(f'{"".join(trg)}\n')
                cnt += 1
        self.logger.info(f'finished decoding {cnt} {mode} instance')

    def update_lr_and_stop_early(self, epoch_idx, devloss, estop):
        prev_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(devloss)
        curr_lr = self.optimizer.param_groups[0]['lr']

        stop_early = True
        if (self.last_devloss - devloss) < estop and \
            prev_lr == curr_lr == self.min_lr:
            self.logger.info(
                'Early stopping triggered with epoch %d (previous dev loss: %f, current: %f)',
                epoch_idx, self.last_devloss, devloss)
            stop_status = stop_early
        else:
            stop_status = not stop_early
        self.last_devloss = devloss
        return stop_status

    def save_model(self, epoch_idx, devloss, eval_res, model_fp):
        eval_tag = '.'.join(['{}_{}'.format(e.desc, e.res) for e in eval_res])
        fp = model_fp + '.nll_{:.4f}.{}.epoch_{}'.format(
            devloss, eval_tag, epoch_idx)
        torch.save(self.model, open(fp, 'wb'))
        self.models.append((fp, devloss, eval_res))

    def reload_and_test(self, model_fp, batch_size, best_acc):
        best_fp, _, best_res = self.models[0]
        best_acc_fp, _, best_acc = self.models[0]
        best_devloss_fp, best_devloss, _ = self.models[0]
        for fp, devloss, res in self.models:
            # [acc, edit distance ]
            if res[0].res >= best_res[0].res and res[1].res <= best_res[1].res:
                best_fp, best_res = fp, res
            if res[0].res >= best_acc[0].res:
                best_acc_fp, best_acc = fp, res
            if devloss <= best_devloss:
                best_devloss_fp, best_devloss = fp, devloss
        self.model = None
        if best_acc:
            best_fp = best_acc_fp
        self.logger.info(f'loading {best_fp} for testing')
        self.load_model(best_fp)
        self.logger.info('decoding dev set')
        self.decode(DEV, f'{model_fp}.decode')
        if self.data.test_file is not None:
            self.calc_loss(TEST, batch_size)
            self.logger.info('decoding test set')
            self.decode(TEST, f'{model_fp}.decode')
            results = self.evaluate(TEST)
            results = ' '.join([f'{r.desc} {r.res}' for r in results])
            self.logger.info(f'TEST {model_fp.split("/")[-1]} {results}')
        return set([best_fp])

    def cleanup(self, saveall, save_fps, model_fp):
        if not saveall:
            for fp, _, _ in self.models:
                if fp in save_fps:
                    continue
                os.remove(fp)
        os.remove(f'{model_fp}.progress')


class Multitrainer:
    INSTANCE_PARAMS = {
        "langs": set(),
        "loglevel": "info",
        "seed": 0,
        "shuffle": True,
        "embed_dim": 20,
        "src_hs": 400,
        "trg_hs": 400,
        "dropout": 0.4,
        "src_layer": 2,
        "trg_layer": 1,
        "max_norm": 5,
        "arch": "hard",
        "estop": 1e-8,
        "pretrain_epochs": 1000,
        "train_epochs": 1000,
        "bs": 20,
        "patience": 10,
        "wid_siz": 11,
        "mono": False,
        "optimizer": "Adam",
        "lr": 1e-3,
        "momentum": 0.9,
        "min_lr": 1e-5,
        "cooldown": 0,
        "discount_factor": .5,
        "bestacc": False,
        "saveall": False
    }

    def __init__(self, **kwargs):
        # a shortcut way of setting a ton of instance variables
        self.__dict__.update(Multitrainer.INSTANCE_PARAMS)
        assert(set(kwargs.keys()).issubset(set(Multitrainer.INSTANCE_PARAMS.keys())))
        self.__dict__.update(kwargs)

        self.src_vocab, self.trg_vocab = dataloader.build_global_vocab(self.langs)

    def train_pair(self, source_lang, target_lang, test=None):
        logger = util.get_logger(os.path.join("logs", f'{source_lang}-{target_lang}.log'), log_level=self.loglevel)
        logger.info(f"Source lang: {source_lang}, Target lang: {target_lang}")

        assert(source_lang is None or source_lang in self.langs)
        assert(target_lang in self.langs)
        pretrain_model_dir = f"model/tag-{self.arch}/{source_lang}" if source_lang else None
        train_model_dir = pretrain_model_dir + "-" + target_lang if source_lang else f"model/tag-{self.arch}/none-{target_lang}"

        if source_lang and not os.path.exists(pretrain_model_dir):
            os.makedirs(pretrain_model_dir)
        if not os.path.exists(train_model_dir):
            os.makedirs(train_model_dir)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        trainer = Trainer(logger)
        trainer.load_data(source_lang=source_lang, target_lang=target_lang, test=test, shuffle=self.shuffle,
                          src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)
        trainer.setup_evalutator()

        pretrain_model_filenames = [f for f in os.listdir(pretrain_model_dir) if re.fullmatch(r"model\.nll.+epoch_\d+", f)] if source_lang else None
        train_model_filenames = [f for f in os.listdir(train_model_dir) if re.fullmatch(r"model\.nll.+epoch_\d+", f)]

        if len(train_model_filenames) > 0:
            print(train_model_filenames)
            train_model_filenames.sort(key=lambda filename: int(filename.split("_")[-1]))
            furthest_model = train_model_filenames[-1]
            pretrain_start_epoch = self.pretrain_epochs
            train_start_epoch = trainer.smart_load_model(os.path.join(train_model_dir, "model")) + 1

            logger.info('continue training from epoch %d', train_start_epoch)
            trainer.setup_training(self.optimizer, self.lr, self.momentum)
            trainer.setup_scheduler(self.min_lr, self.patience, self.cooldown, self.discount_factor)
            trainer.load_training(train_model_dir)

        elif source_lang and len(pretrain_model_filenames) > 0:
            pretrain_model_filenames.sort(key=lambda filename: int(filename.split("_")[-1]))
            furthest_model = pretrain_model_filenames[-1]
            pretrain_start_epoch = trainer.smart_load_model(os.path.join(pretrain_model_dir, "model")) + 1
            train_start_epoch = 0

            logger.info('continue pretraining from epoch %d', pretrain_start_epoch)
            trainer.setup_training(self.optimizer, self.lr, self.momentum)
            trainer.setup_scheduler(self.min_lr, self.patience, self.cooldown, self.discount_factor)
            trainer.load_training(pretrain_model_dir)

        else:
            print(source_lang)
            print(os.listdir(pretrain_model_dir))
            print([re.fullmatch(r"model\.nll.+epoch_\d+", f) for f in os.listdir(pretrain_model_dir)])
            print(pretrain_model_filenames)
            logger.info("Creating model from scratch")
            pretrain_start_epoch = 0 if source_lang else self.pretrain_epochs
            train_start_epoch = 0
            trainer.build_model(
                embed_dim=self.embed_dim,
                dropout=self.dropout,
                src_hs=self.src_hs,
                trg_hs=self.trg_hs,
                src_layer=self.src_layer,
                trg_layer=self.trg_layer,
                wid_siz=self.wid_siz,
                arch=self.arch,
                mono=self.mono)
            trainer.setup_training(self.optimizer, self.lr, self.momentum)
            trainer.setup_scheduler(self.min_lr, self.patience, self.cooldown, self.discount_factor)

        for epoch_idx in range(pretrain_start_epoch, self.pretrain_epochs):
            trainer.train(epoch_idx, self.bs, self.max_norm, pre=True)
            with torch.no_grad():
                devloss = trainer.calc_loss(DEV, self.bs, epoch_idx)
                eval_res = trainer.evaluate(DEV, epoch_idx)
            if trainer.update_lr_and_stop_early(epoch_idx, devloss, self.estop):
                break
            fp = os.path.join(pretrain_model_dir, "model")
            trainer.save_model(epoch_idx, devloss, eval_res, fp)
            trainer.save_training(fp)

        for epoch_idx in range(train_start_epoch, self.train_epochs):
            trainer.train(epoch_idx, self.bs, self.max_norm, pre=False)
            with torch.no_grad():
                devloss = trainer.calc_loss(DEV, self.bs, epoch_idx)
                eval_res = trainer.evaluate(DEV, epoch_idx)
            if trainer.update_lr_and_stop_early(epoch_idx, devloss, self.estop):
                break
            fp = os.path.join(train_model_dir, "model")
            trainer.save_model(epoch_idx, devloss, eval_res, fp)
            trainer.save_training(fp)

        with torch.no_grad():
            save_fps = trainer.reload_and_test(os.path.join(train_model_dir, "model"), self.bs, self.bestacc)

        # trainer.cleanup(saveall, save_fps, os.path.join(train_model_dir, "model"))
