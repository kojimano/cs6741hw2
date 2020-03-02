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
            print(total_samples)
            for j in range(len(seq) - n_gram + 1):
                input = tuple([seq[j + k] for k in range(model._n-1)])
                prob = model(input)
                logit = prob.log()
                target = torch.LongTensor([seq[j+model._n-1]])
                loss = criterion(logit.unsqueeze(0),target)
                total_loss += loss

                total_samples += 1
        if total_samples > 5000:
            break
    avg_loss = total_loss / total_samples
    avg_perplexity = torch.exp(avg_loss)
    avg_loss = avg_loss.item()
    avg_perplexity = avg_perplexity.item()
    print("Evaluate, loss: {}, perplexity: {}".format(avg_loss, avg_perplexity))
    return avg_perplexity

def run_em(model: nn.Module, val_iter: torchtext.datasets, test_iter: torchtext.datasets):
    max_turn = 10
    max_samples = 500
    n_gram = model._n
    alpha1 = 1/3.
    alpha2 = 1/3.
    alpha3 = 1 - alpha1 - alpha2

    for turn in range(max_turn):
        # Miscs
        all_sents_log_prob_alpha1 = []
        all_sents_log_prob_alpha2 = []
        all_sents_log_prob_alpha3 = []
        total_samples = 0
        total_probs = []
        all_targets = []

        # Expectation
        for batch in iter(val_iter):
            for i in range(batch.text.shape["batch"]):
                seq = [i for i in batch.text[{"batch": i}].tolist()]
                for j in range(len(seq) - n_gram + 1):
                    input = tuple([seq[j + k] for k in range(model._n-1)])
                    prob = model(input, em=True)
                    target = torch.LongTensor([seq[j+model._n-1]])
                    all_targets.append(target.item())
                    total_probs.append(prob.numpy())
                    total_samples += 1
                if total_samples > max_samples:
                    break
            if total_samples > max_samples:
                break

        # Maximization
        total_probs = np.stack(total_probs, 2)
        all_targets = np.stack(all_targets, 0)
        target_probs = total_probs[all_targets,:,[i for i in range(len(all_targets))]]
        prob_alpha1 = np.mean(alpha1 * target_probs[:,0])
        prob_alpha2 = np.mean(alpha2 * target_probs[:,1])
        prob_alpha3 = np.mean(alpha3 * target_probs[:,2])

        alpha1, alpha2, alpha3 = prob_alpha1, prob_alpha2, prob_alpha3
        norm = (alpha1 + alpha2 + alpha3)
        alpha1 = alpha1 / norm
        alpha2 = alpha2 / norm
        alpha3 = alpha3 / norm
        model.set_alpha(alpha1, alpha2, alpha3)
        print(alpha1, alpha2, alpha3)


# TODO maybe worng model ... retraining can be option
def analysis(train_iter):

    import torch.nn.functional as F
    # Get models
    tg_model = get_model("tg", train_iter, None)
    nn_model = get_model("nn", train_iter, None)
    nn_model.load_state_dict(torch.load("./checkpoints/best-nn-score-326.637-epoch-49.pkl"))
    lstm_model = get_model("lstm", train_iter, None)
    lstm_model.load_state_dict(torch.load("./checkpoints/best-lstm-score-154.297-epoch-49.pkl"))
    if torch.cuda.is_available():
        tg_model.cuda()
        nn_model.cuda()
        lstm_model.cuda()
    tg_model.eval()
    nn_model.eval()
    lstm_model.eval()

    # Get train context
    print(train_txt.examples[0].text[93:104])
    context1 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in train_txt.examples[0].text[93:104]]).long().cuda().unsqueeze(0)
    print(train_txt.examples[0].text[210:221])
    context2 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in train_txt.examples[0].text[210:221]]).long().cuda().unsqueeze(0)
    print(test_txt.examples[0].text[12:24])
    context3 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in test_txt.examples[0].text[12:24]]).long().cuda().unsqueeze(0)
    print(test_txt.examples[0].text[220:231])
    context4 = torch.Tensor([TEXT.vocab.stoi[tk] for tk in test_txt.examples[0].text[220:231]]).long().cuda().unsqueeze(0)
    contexts = [context1, context2, context3, context4]

    # Get test context
    KL_loss = nn.KLDivLoss()
    for context in contexts:
        context.cuda()
        max_idx = context[0][-1].item()
        print("Ground-truth {}".format(TEXT.vocab.itos[max_idx]))
        # Trigram
        tg_probs = tg_model(tuple([context[0][-2].item(), context[0][-1].item()]))
        tg_probs = tg_probs.cuda().unsqueeze(0)
        max_idx = torch.argmax(tg_probs)
        print("Pred: {}, Entophy {}".format(TEXT.vocab.itos[max_idx], Categorical(tg_probs).entropy()))

        # NN
        nn_probs = nn_model(context[0][-6:-1].unsqueeze(0))[:,-1,:]
        nn_probs = F.softmax(nn_probs)
        max_idx = torch.argmax(nn_probs)
        print("Pred: {}, Entophy {}".format(TEXT.vocab.itos[max_idx], Categorical(nn_probs).entropy() ))

        # LSTM
        lstm_probs = lstm_model(context[0][:-1].unsqueeze(0))[:,-1,:]
        lstm_probs = F.softmax(lstm_probs)
        max_idx = torch.argmax(lstm_probs)
        print("Pred: {}, Entophy {}".format(TEXT.vocab.itos[max_idx], Categorical(lstm_probs).entropy()))

        # KL divergence
        print(KL_loss(nn_probs.log(), lstm_probs))
        print(KL_loss(nn_probs.log(), tg_probs))
        print(KL_loss(lstm_probs.log(), tg_probs))




if __name__=="__main__":
    args = parser.parse_args()
    train_iter, val_iter, test_iter = LMDataset.splits(
        (train_txt, val_txt, test_txt), batch_size=10, device=torch.device("cuda"), bptt_len=32, repeat=False)
    if args.analysis:
        analysis(train_iter)
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
