'''
Decode model
'''
import argparse
from functools import partial

import torch

from .dataloader import BOS, EOS, UNK_IDX, read_file
from .model import decode_beam_search, decode_greedy
from .util import maybe_mkdir
from .train import get_model_dirs, get_furthest_model


# def get_args():
#     # yapf: disable
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--in_file', required=True, help='Dev/Test file')
#     parser.add_argument('--out_file', required=True, help='Output file')
#     parser.add_argument('--lang', required=True, help='Language tag')
#     parser.add_argument('--model', required=True, help='Path to model')
#     parser.add_argument('--max_len', default=100, type=int)
#     parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'])
#     parser.add_argument('--beam_size', default=5, type=int)
#     parser.add_argument('--nonorm', default=False, action='store_true')
#     return parser.parse_args()
#     # yapf: enable


def setup_inference(decode="greedy", max_len=100, beam_size=5, nonorm=False):
    decode_fn = None
    if decode == 'greedy':
        decode_fn = partial(decode_greedy, max_len=max_len)
    elif decode == 'beam':
        decode_fn = partial(
            decode_beam_search,
            max_len=max_len,
            nb_beam=beam_size,
            norm=not nonorm)
    else:
        raise ValueError
    return decode_fn


def encode(model, lemma, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in lemma:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    attr = [0] * (len(model.attr_c2i) + 1)
    for tag in tags:
        if tag in model.attr_c2i:
            attr_idx = model.attr_c2i[tag] - tag_shift
        else:
            attr_idx = -1
        if attr[attr_idx] == 0:
            attr[attr_idx] = model.attr_c2i.get(tag, 0)

    return (torch.tensor(src, device=device).view(len(src), 1),
            torch.tensor(attr, device=device).view(1, len(attr)))


def main(arch, source_lang, target_lang, test=True):
    decode_fn = setup_inference()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_dir = get_model_dirs(arch, source_lang, target_lang)
    furthest_model, _ = get_furthest_model(model_dir)
    model = torch.load(open(furthest_model, mode='rb'), map_location=device)
    model = model.to(device)

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]

    test_file = f"conll2018/task1/all/{target_lang}-{'test' if test else 'train-low'}"
    out_file = f"{model_dir}/predictions.txt"

    maybe_mkdir(out_file)
    total_guesses = 0
    correct_guesses = 0
    with open(out_file, 'w', encoding='utf-8') as fp:
        for lemma, word, tags in read_file(test_file):
            src = encode(model, lemma, tags, device)
            pred, _ = decode_fn(model, src)
            pred_out = ''.join(decode_trg(pred))
            fp.write(f'{"".join(lemma)}\t{pred_out}\t{";".join(tags[1:])}\n')

            total_guesses += 1
            if ''.join(word) == pred_out:
                correct_guesses += 1

    print("accuracy", f"{round(correct_guesses*100/total_guesses, 2)}%")
    return correct_guesses, total_guesses
