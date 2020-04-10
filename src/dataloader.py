import torch
import numpy as np
import os

from tqdm import tqdm

from code.timer import Timer
train_timer = Timer.get("train")

BOS = '<s>'
EOS = '<\s>'
PAD = '<PAD>'
UNK = '<UNK>'
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def read_file(file):
    if 'train' in file:
        lang_tag = [file.split('/')[-1].split('-train')[0]]
    elif 'dev' in file:
        lang_tag = [file.split('/')[-1].split('-dev')[0]]
    else:
        raise ValueError
    with open(file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            lemma, word, tags = line.strip().split('\t')
            yield list(lemma), list(word), lang_tag + tags.split(';')


def build_global_vocab(langs):
    char_set, tag_set = set(), set()

    for lang in langs:
        files = [os.path.join("conll2018/task1/all", file) for file in
                 [f"{lang}-train-high", f"{lang}-train-low"]  # , f"{lang}-dev"]
                 if file in os.listdir("conll2018/task1/all")]
        for file in files:
            for lemma, word, tags in read_file(file):
                char_set.update(lemma)
                char_set.update(word)
                tag_set.update(tags)

    chars = sorted(list(char_set))
    tags = sorted(list(tag_set))
    source = [PAD, BOS, EOS, UNK] + chars + tags
    target = [PAD, BOS, EOS, UNK] + chars

    return source, target


class TagSIGMORPHON2019Task1:
    def __init__(self, source_lang, target_lang, src_vocab, trg_vocab, test=True, shuffle=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # assert os.path.isfile(train_file)
        # assert os.path.isfile(dev_file)
        # assert test_file is None or os.path.isfile(test_file)
        self.pretrain_file = f"conll2018/task1/all/{source_lang}-train-high" if source_lang else None
        self.train_file = f"conll2018/task1/all/{target_lang}-train-low"
        self.dev_file = f"conll2018/task1/all/{target_lang}-dev"
        self.test_file = f"conll2018/task1/all/{target_lang}-test" if test else None
        self.shuffle = shuffle
        self.batch_data = dict()
        self.source, self.target = src_vocab, trg_vocab
        self.file_sizes()
        self.nb_attr = len(src_vocab) - len(trg_vocab)
        self.source_vocab_size = len(self.source)
        self.target_vocab_size = len(self.target)
        if self.nb_attr > 0:
            self.source_c2i = {
                c: i
                for i, c in enumerate(self.source[:-self.nb_attr])
            }
            self.attr_c2i = {
                c: i + len(self.source_c2i)
                for i, c in enumerate(self.source[-self.nb_attr:])
            }
        else:
            self.source_c2i = {c: i for i, c in enumerate(self.source)}
            self.attr_c2i = None
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.sanity_check()

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK

    def _batch_sample(self, batch_size, file):
        if isinstance(file, list):
            key = tuple(sorted(file))
        else:
            key = file
        if key not in self.batch_data:
            lst = list()
            for t in self._iter_helper(file):
                lst.append((len(t[0]), *t))
            print("made new batch data: ", key)
            self.batch_data[key] = sorted(lst, key=lambda x: x[0])

        lst = self.batch_data[key]

        if self.shuffle:
            lst = np.random.permutation(lst)
        for start in range(0, len(lst), batch_size):
            yield self._batch_helper(lst[start:start + batch_size])

    def pretrain_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.pretrain_file)

    def train_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.train_file)

    def dev_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.dev_file)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.test_file)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        l = len(sent)
        s = []
        for x in sent:
            if x in self.source_c2i:
                s.append(self.source_c2i[x])
            else:
                s.append(self.attr_c2i[x])
        return torch.tensor(s, device=self.device).view(l, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]

    def file_sizes(self):
        if self.pretrain_file is None:
            self.nb_pretrain = 0
        else:
            self.nb_pretrain = sum([1 for _ in read_file(self.pretrain_file)])
        self.nb_train = sum([1 for _ in read_file(self.train_file)])
        self.nb_dev = sum([1 for _ in read_file(self.dev_file)])
        if self.test_file is None:
            self.nb_test = 0
        else:
            self.nb_test = sum([1 for _ in read_file(self.test_file)])

    def _iter_helper(self, file):
        tag_shift = len(self.source) - self.nb_attr
        if not isinstance(file, list):
            file = [file]
        for fp in file:
            i = 0
            for lemma, word, tags in read_file(fp):
                i += 1
                src = []
                src.append(self.source_c2i[BOS])
                for char in lemma:
                    src.append(self.source_c2i.get(char, UNK_IDX))
                src.append(self.source_c2i[EOS])
                trg = []
                trg.append(self.target_c2i[BOS])
                for char in word:
                    trg.append(self.target_c2i.get(char, UNK_IDX))
                trg.append(self.target_c2i[EOS])
                attr = [0] * (self.nb_attr + 1)
                for tag in tags:
                    if tag in self.attr_c2i:
                        attr_idx = self.attr_c2i[tag] - tag_shift
                    else:
                        attr_idx = -1
                    if attr[attr_idx] == 0:
                        attr[attr_idx] = self.attr_c2i.get(tag, 0)
                yield src, trg, attr
            print(fp, "had", i, "examples")

    def _batch_helper(self, lst):
        train_timer.task("batching")
        bs = len(lst)
        srcs, trgs, attrs = [], [], []
        max_src_len, max_trg_len, max_nb_attr = 0, 0, 0
        for _, src, trg, attr in lst:
            max_src_len = max(len(src), max_src_len)
            max_trg_len = max(len(trg), max_trg_len)
            max_nb_attr = max(len(attr), max_nb_attr)
            srcs.append(src)
            trgs.append(trg)
            attrs.append(attr)
        batch_attr = torch.zeros(
            (bs, max_nb_attr), dtype=torch.long, device=self.device)
        batch_src = torch.zeros(
            (max_src_len, bs), dtype=torch.long, device=self.device)
        batch_src_mask = torch.zeros(
            (max_src_len, bs), dtype=torch.float, device=self.device)
        batch_trg = torch.zeros(
            (max_trg_len, bs), dtype=torch.long, device=self.device)
        batch_trg_mask = torch.zeros(
            (max_trg_len, bs), dtype=torch.float, device=self.device)
        for i in range(bs):
            for j in range(len(attrs[i])):
                batch_attr[i, j] = attrs[i][j]
            for j in range(len(srcs[i])):
                batch_src[j, i] = srcs[i][j]
                batch_src_mask[j, i] = 1
            for j in range(len(trgs[i])):
                batch_trg[j, i] = trgs[i][j]
                batch_trg_mask[j, i] = 1
        return ((batch_src, batch_attr), batch_src_mask, batch_trg,
                batch_trg_mask)

    def pretrain_sample(self):
        for src, trg, tags in self._iter_helper(self.pretrain_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def train_sample(self):
        for src, trg, tags in self._iter_helper(self.train_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def dev_sample(self):
        for src, trg, tags in self._iter_helper(self.dev_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))

    def test_sample(self):
        for src, trg, tags in self._iter_helper(self.test_file):
            yield ((torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags))),
                   torch.tensor(trg, device=self.device).view(len(trg), 1))
