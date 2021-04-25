import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import transformers
from torch.nn.modules.loss import _WeightedLoss
from tqdm.autonotebook import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


class CFG:
    LEARNING_RATE = 4e-5
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 3
    PATIENCE = 2
    INPUT_DIR = "../input"
    OUTPUT_DIR = "../output"
    TRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "roberta_tpu")
    TRAINING_FILE = os.path.join(INPUT_DIR, "tweet-train-folds-v2/train_folds.csv")
    ROBERTA_PATH = os.path.join(INPUT_DIR, "roberta-base")
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=os.path.join(ROBERTA_PATH, "vocab.json"),
        merges_file=os.path.join(ROBERTA_PATH, "merges.txt"),
        lowercase=True,
        add_prefix_space=True,
    )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind : ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1:offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    return {
        "ids": input_ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "targets_start": targets_start,
        "targets_end": targets_end,
        "orig_tweet": tweet,
        "orig_selected": selected_text,
        "sentiment": sentiment,
        "offsets": tweet_offsets,
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tokenizer, max_len):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], self.selected_text[item], self.sentiment[item], self.tokenizer, self.max_len
        )

        return {
            "ids": torch.tensor(data["ids"], dtype=torch.long),
            "mask": torch.tensor(data["mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(data["token_type_ids"], dtype=torch.long),
            "targets_start": torch.tensor(data["targets_start"], dtype=torch.long),
            "targets_end": torch.tensor(data["targets_end"], dtype=torch.long),
            "orig_tweet": data["orig_tweet"],
            "orig_selected": data["orig_selected"],
            "sentiment": data["sentiment"],
            "offsets": torch.tensor(data["offsets"], dtype=torch.long),
        }


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, roberta_path, model_config):
        super(TweetModel, self).__init__(model_config)
        self.roberta = transformers.RobertaModel.from_pretrained(roberta_path, config=model_config)
        self.dropout = nn.Dropout(0.3)

        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output, hidden_states = self.roberta(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        out = torch.sum(torch.stack(hidden_states[-4:]), 0) / 4
        out = self.dropout(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            pass

        return loss


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = SmoothCrossEntropyLoss(smoothing=0.15)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        start_logits, end_logits = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(start_logits, end_logits, targets_start, targets_end, device)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].cpu().numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            start_logits, end_logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end, device)

            start_preds = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                filtered_output, _, _, _ = get_sentence(
                    tweet=tweet, start_preds=start_preds[px, :], end_preds=end_preds[px, :], offsets=offsets[px]
                )
                selected_tweet = orig_selected[px]
                jaccard_score = jaccard(selected_tweet, filtered_output)
                jaccard_scores.append(jaccard_score)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=loss.item())
    return jaccards.avg


def get_best_start_end_idxs(start_preds, end_preds):
    best_pred = -1000
    best_idxs = None
    for start_idx, start_pred in enumerate(start_preds):
        for end_idx, end_pred in enumerate(end_preds[start_idx:]):
            pred_sum = (start_pred + end_pred).item()
            if pred_sum > best_pred:
                best_pred = pred_sum
                best_idxs = (start_idx, start_idx + end_idx)
    return best_idxs, best_pred / 2


def get_sentence(tweet, start_preds, end_preds, offsets):
    (idx_start, idx_end), best_pred = get_best_start_end_idxs(start_preds, end_preds)

    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += tweet[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "
    return filtered_output, idx_start, idx_end, best_pred


def run(fold):
    model_config = transformers.RobertaConfig.from_pretrained(CFG.ROBERTA_PATH)
    model_config.output_hidden_states = True
    MX = TweetModel(CFG.ROBERTA_PATH, model_config)

    dfx = pd.read_csv(CFG.TRAINING_FILE)
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = xm.xla_device(fold + 1)
    model = MX.to(device)

    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tokenizer=CFG.TOKENIZER,
        max_len=CFG.MAX_LEN,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.TRAIN_BATCH_SIZE,
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        tokenizer=CFG.TOKENIZER,
        max_len=CFG.MAX_LEN,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.VALID_BATCH_SIZE,
    )

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    num_train_steps = int(len(df_train) / CFG.TRAIN_BATCH_SIZE * CFG.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=CFG.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    es = EarlyStopping(patience=CFG.PATIENCE, mode="max")

    for epoch in range(CFG.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        jac = eval_fn(valid_data_loader, model, device)
        print(f"Fold={fold}, Epoch={epoch}, Jaccard={jac}")
        es(jac, model, model_path=os.path.join(CFG.TRAINED_MODEL_PATH, f"model_{fold}.bin"))
        if es.early_stop:
            print("Early stopping")
            print(f"Fold={fold} Epoch={epoch}, BestScore={es.best_score}")
            return es.best_score, fold
        if epoch == (CFG.EPOCHS - 1):
            print(f"Fold={fold} Epoch={epoch}, BestScore={es.best_score}")
            return es.best_score, fold
