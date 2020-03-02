# python main.py --model_type tg
import os, time
import operator
import argparse
import numpy as np
from IPython import embed

import torch
import torchtext
import torch.nn as nn
from torch.distributions import Categorical
from namedtensor import ntorch
from namedtensor.text import NamedField
from torchtext.vocab import Vectors

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset
import math

import models.models as models

# parser
parser = argparse.ArgumentParser(description='CS_6741_HW_2')
parser.add_argument('--model_type', default='', type=str, help='tg | nn | lstm')
parser.add_argument('--pretrained', default='', type=str, help='model path')
parser.add_argument('--em', action='store_true')
parser.add_argument('--analysis', action='store_true')
parser.add_argument('--use_word_vec', action='store_true')


TEXT = NamedField(names=("seqlen",))

train_txt, val_txt, test_txt = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

TEXT.build_vocab(train_txt)
url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
word_vec = TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


class LMDataset(BPTTIterator):
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = (data
            .stack(("seqlen", "batch"), "flat")
            .split("flat", ("batch", "seqlen"), batch=self.batch_size)
            .transpose("seqlen", "batch")
        )

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text = data.narrow("seqlen", i, seq_len),
                    target = data.narrow("seqlen", i+1, seq_len),
                )

            if not self.repeat:
                return

def get_model(model_type: str, train_iter: torchtext.data.BucketIterator, word_vec:torch.Tensor) -> nn.Module:
    if model_type == "tg":
        model = models.TrigramLM(vocab_size=len(TEXT.vocab))
        model.set_ngrams(train_iter)
    elif model_type == "nn":
        model = models.NeuralNetLM(vocab_size=len(TEXT.vocab))
    elif model_type == "lstm":
        model = models.LstmLM(vocab_size=len(TEXT.vocab), word_vec=word_vec)
    else:
        raise ValueError("Not supported model type.")

    if torch.cuda.is_available():
        model.cuda()

    return model

def calc_sent_perplexity(probs, sent_len):
    all_perplexity = torch.ones(probs.shape[0])
    if torch.cuda.is_available():
        all_perplexity = all_perplexity.cuda()

    for seq_idx in range(probs.shape[1]):
        all_perplexity *= (probs[:, seq_idx]  ** (-1/sent_len))

    return all_perplexity

def calc_sent_log_probability(probs):
    all_probs = torch.zeros(probs.shape[0])
    all_probs = all_probs.cuda()

    for seq_idx in range(probs.shape[1]):
        all_probs += torch.log(probs[:, seq_idx])

    return all_probs


def save_model(model: nn.Module, model_name: str):
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(model.state_dict(), "./checkpoints/"+model_name)


def train(model: nn.Module,  train_iter: torchtext.data.BucketIterator, val_iter: torchtext.datasets, args):

    if args.model_type == "nn":
        model_name = "nn"
        no_bias = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bias)],
                "weight_decay": 1e-5,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bias)], "weight_decay": 0.0},
        ]

        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=2e-2)
        criterion = torch.nn.CrossEntropyLoss()

    elif args.model_type == "lstm":
        model_name = "lstm"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("model type: {} not supported.".format(args.model_type))

    # training parameters
    num_epchs = 50
    log_step = 20
    model_val_scores = {}

    for epoch in range(num_epchs):
        epoch_loss = 0
        model.train()
        for i, batch in enumerate(train_iter):
            model.zero_grad()
            logits = model(batch.text.values)
            logits = logits.contiguous()
            logits = logits.view(-1, model._vocab_size)
            labels = batch.target.values.flatten()
            loss = criterion(logits, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                avg_loss = epoch_loss / (i + 1)
                avg_loss = avg_loss.item()
                print("Epoch {}, step {}, average loss: {}".format(epoch, i, avg_loss))

        avg_loss = epoch_loss / len(train_iter)
        avg_perplexity = torch.exp(avg_loss)
        avg_loss = avg_loss.item()
        avg_perplexity = avg_perplexity.item()
        print("Epoch {}, loss: {}, perplexity: {}".format(epoch, avg_loss, avg_perplexity))

        # Validation
        eval_perplexity = test(model, val_iter)

        # Save model
        eval_perplexity = np.round(eval_perplexity, 3)
        pkl_name = model_name + "-score-{}-epoch-{}.pkl".format(eval_perplexity, epoch)
        save_model(model, pkl_name)
        model_val_scores[pkl_name] = eval_perplexity

    sorted_models = sorted(model_val_scores.items(), key=operator.itemgetter(1), reverse=False)
    for pkl_name, _ in sorted_models[1:]:
            os.remove("./checkpoints/{}".format(pkl_name))
    final_model_name = "best-{}".format(sorted_models[0][0])
    os.rename("./checkpoints/{}".format(sorted_models[0][0]), "./checkpoints/{}".format(final_model_name))

    return "./checkpoints/" + final_model_name

def test(model: nn.Module, test_iter: torchtext.data.BucketIterator):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0
    for batch in test_iter:
        with torch.no_grad():
            logits = model(batch.text.values)
            logits = logits.contiguous()
            logits = logits.view(-1, model._vocab_size)
            labels = batch.target.values.flatten()
            loss = criterion(logits, labels)
            epoch_loss += loss

    avg_loss = epoch_loss / len(test_iter)
    avg_perplexity = torch.exp(avg_loss)
    avg_loss = avg_loss.item()
    avg_perplexity = avg_perplexity.item()
    print("Evaluate, loss: {}, perplexity: {}".format(avg_loss, avg_perplexity))
    return avg_perplexity


def test_trigram(model: models.TrigramLM, test_iter: torchtext.data.BucketIterator):
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    log_step = 100
    n_gram = model._n

    for batch in iter(val_iter):
        for i in range(batch.text.shape["batch"]):
            seq = [i for i in batch.text[{"batch": i}].tolist()]
            print(i)
            for j in range(len(seq) - n_gram + 1):
                input = tuple([seq[j + k] for k in range(model._n-1)])
                prob = model(input)
                logit = prob.log()
                target = torch.LongTensor([seq[j+model._n-1]])
                loss = criterion(logit.unsqueeze(0),target)
                total_loss += loss
                total_samples += 1
    embed()
    avg_loss = total_loss / len(total_samples)
    avg_perplexity = torch.exp(avg_loss)
    avg_loss = avg_loss.item()
    avg_perplexity = avg_perplexity.item()
    print("Evaluate, loss: {}, perplexity: {}".format(avg_loss, avg_perplexity))
    return avg_perplexity

def run_em(model: nn.Module, val_iter: torchtext.datasets, test_iter: torchtext.datasets):
    max_turn = 3
    alpha1 = 1/3.
    alpha2 = 1/3.
    alpha3 = 1 - alpha1 - alpha2

    for turn in range(max_turn):
        # Prplexity calculation
        print("Turn {}".format(turn))
        print(alpha1, alpha2, alpha3)
        test(model, test_iter)

        # Miscs
        all_sents_log_prob_alpha1 = []
        all_sents_log_prob_alpha2 = []
        all_sents_log_prob_alpha3 = []
        # Maximization
        for idx, batch in enumerate(val_iter):
            batch.text = batch.text.t()
            probs_alpha1, probs_alpha2, probs_alpha3 = model.em_forward(batch.text)
            sents_probs_alpha1 = calc_sent_log_probability(probs_alpha1.cuda())
            sents_probs_alpha2 = calc_sent_log_probability(probs_alpha2.cuda())
            sents_probs_alpha3 = calc_sent_log_probability(probs_alpha3.cuda())

            all_sents_log_prob_alpha1.append(sents_probs_alpha1)
            all_sents_log_prob_alpha2.append(sents_probs_alpha2)
            all_sents_log_prob_alpha3.append(sents_probs_alpha3)

        # Expectation
        all_sents_prob_alpha1 = torch.cat(all_sents_log_prob_alpha1, 0)
        all_sents_prob_alpha2 = torch.cat(all_sents_log_prob_alpha2, 0)
        all_sents_prob_alpha3 = torch.cat(all_sents_log_prob_alpha3, 0)

        all_sents_prob_alpha1 = all_sents_prob_alpha1 - torch.mean(all_sents_prob_alpha3)
        all_sents_prob_alpha2 = all_sents_prob_alpha2 - torch.mean(all_sents_prob_alpha3)
        all_sents_prob_alpha3 = all_sents_prob_alpha3 - torch.mean(all_sents_prob_alpha3)

        alpha1 = alpha1 * torch.sum(torch.exp(all_sents_prob_alpha1))
        alpha2 = alpha2 * torch.sum(torch.exp(all_sents_prob_alpha2))
        alpha3 = alpha3 * torch.sum(torch.exp(all_sents_prob_alpha3))
        alpha1 = alpha1 / (alpha1 + alpha2 + alpha3)
        alpha2 = alpha2 / (alpha1 + alpha2 + alpha3)
        alpha3 = alpha3 / (alpha1 + alpha2 + alpha3)
        model.set_alpha(alpha1.item(), alpha2.item(), alpha3.item())


def analysis(train_txt, test_txt, word_vec):

    import torch.nn.functional as F

    tg_model = get_model("tg", train_txt, None)
    nn_model = get_model("nn", train_txt, None)
    nn_model.load_state_dict(torch.load("./checkpoints/best-nn-score-614.304-epoch-29.pkl"))
    lstm_model = get_model("lstm", train_txt, None)
    lstm_model.load_state_dict(torch.load("./checkpoints/best-lstm-score-7484.329-epoch-0.pkl"))
    tg_model.cuda()
    nn_model.cuda()
    lstm_model.cuda()
    tg_model.eval()
    nn_model.eval()
    lstm_model.eval()

    print(train_txt.examples[0].text[97:104])
    train_context1 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in train_txt.examples[0].text[97:104]]).long().cuda().unsqueeze(0)
    print(train_txt.examples[0].text[95:100])
    train_context2 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in train_txt.examples[0].text[101:109]]).long().cuda().unsqueeze(0)
    print(train_txt.examples[0].text[95:100])
    train_context3 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in train_txt.examples[0].text[150:154]]).long().cuda().unsqueeze(0)

    train_examples = torch.cat([train_context1, train_context2, train_context3], 0)

    test_context1 = []
    test_context2 = []
    test_context3 = []
    test_examples = [test_context1, test_context2, test_context3]

    train_kl_tables = torch.zeros(3, 3, len(train_examples))
    train_ent_tables = torch.zeros(3, len(train_examples))

    tg_probs = tg_model.analysis(train_examples)
    tg_probs = tg_probs.cuda()
    nn_probs = F.softmax(nn_model(train_examples), 2)[:,-1,:]
    lstm_probs = F.softmax(lstm_model(train_examples), 2)[:,-1,:]
    train_ent_tables[0,i] =  Categorical(tg_probs).entropy()
    train_ent_tables[1,i] = Categorical(nn_probs).entropy()
    train_ent_tables[2,i] = Categorical(lstm_probs).entropy()

    """
    train_kl_tables[0,1,i] = F.kl_div(tg_probs, tg_probs)
    train_kl_tables[0,2,i] = F.kl_div(tg_probs, nn_probs)
    train_kl_tables[1,2,i] = F.kl_div(nn_probs, lstm_probs)
    """


if __name__=="__main__":
    args = parser.parse_args()
    train_iter, val_iter, test_iter = LMDataset.splits(
        (train_txt, val_txt, test_txt), batch_size=10, device=torch.device("cuda"), bptt_len=32, repeat=False)
    if args.analysis:
        analysis(train_txt, test_txt, word_vec)
    else:
        # Model instantiation
        model = get_model(args.model_type, train_iter, word_vec) if args.use_word_vec else get_model(args.model_type, train_iter, None)

        # Train
        if args.pretrained == "" and args.model_type != "tg":
            args.pretrained = train(model, train_iter, val_iter, args)
        elif args.model_type == "tg":
            if args.em:
                run_em(model, val_iter, test_iter)

        # load model
        if args.model_type != "tg":
            model.load_state_dict(torch.load(args.pretrained))

        # Test
        if args.model_type == "tg":
            mean_perplexity = test_trigram(model, test_iter)
        else:
            mean_perplexity = test(model, test_iter)
