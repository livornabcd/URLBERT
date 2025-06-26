import torch
import torch.nn as nn
import torch.nn.functional as F
import options
from dataloader import generate_dataloader
from buildmodel import buildBERT
import random
import numpy as np
from timerecord import format_time
from tqdm import tqdm
from torch.optim import AdamW
import time
from torch.cuda.amp import autocast as autocast, GradScaler
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def shuffle_tokens(input_ids, shuffle_prob=0.15):
    labels = torch.zeros_like(input_ids)
    shuffled = input_ids.clone()
    for i in range(input_ids.size(0)):
        tokens = shuffled[i]
        mask = torch.rand(tokens.size(), device=tokens.device) < shuffle_prob
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            perm = idx[torch.randperm(idx.numel())]
            shuffled[i, idx] = tokens[perm]
            labels[i, idx] = 1
    return shuffled, labels

def replace_tokens(input_ids, replace_prob=0.15, vocab_size=5000):
    labels = torch.zeros_like(input_ids)
    replaced = input_ids.clone()
    mask = torch.rand(input_ids.size(), device=input_ids.device) < replace_prob
    random_tokens = torch.randint(0, vocab_size, input_ids.size(), device=input_ids.device)
    replaced[mask] = random_tokens[mask]
    labels[mask] = 1
    return replaced, labels

def search_path(file, directory):
    count = 0
    for filename in os.listdir(directory):
        if file in filename:
            count += 1
    return count


class Config:
    def __init__(self):
        pass

    def training_config(
            self,
            batch_size,
            epochs,
            learning_rate,
            weight_decay,
            device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device


def train(model, train_dataloader, train_sampler, config, optimizer, scaler, data_step):
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else model.encoder.config.hidden_size
    std_head = nn.Linear(hidden_size, 1).to(config.device)
    rtd_head = nn.Linear(hidden_size, 1).to(config.device)
    optimizer.add_param_group({'params': std_head.parameters()})
    optimizer.add_param_group({'params': rtd_head.parameters()})

    model.train()
    training_loss = []
    time_t0 = time.time()
    for step, batch in enumerate(tqdm(train_dataloader, desc='Step')):
        train_sampler.set_epoch(step)
        input_ids, attention_mask, token_type_ids, mlm_labels = (
            b.long().to(config.device) for b in batch
        )

        std_input, std_labels = shuffle_tokens(input_ids)
        rtd_input, rtd_labels = replace_tokens(input_ids, vocab_size=model.config.vocab_size)

        with autocast():
            out_mlm = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=mlm_labels,
                            output_hidden_states=True)
            mlm_loss = out_mlm[0]

            # STD forward
            out_std = model(std_input,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
            std_logits = std_head(out_std.hidden_states[-1]).squeeze(-1)
            std_loss = F.binary_cross_entropy_with_logits(std_logits, std_labels.float())

            # RTD forward
            out_rtd = model(rtd_input,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
            rtd_logits = rtd_head(out_rtd.hidden_states[-1]).squeeze(-1)
            rtd_loss = F.binary_cross_entropy_with_logits(rtd_logits, rtd_labels.float())

            loss = mlm_loss + config.alpha * std_loss + config.beta * rtd_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + \
                                       list(std_head.parameters()) + \
                                       list(rtd_head.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad()

        training_loss.append(loss.item())
        if (step+1) % 500 == 0:
            print(f"Step {step+1} loss: mlm={mlm_loss:.4f}, std={std_loss:.4f}, rtd={rtd_loss:.4f}")

    avg_loss = sum(training_loss) / len(training_loss)
    print(f"Training loss: {avg_loss:.4f}; step {data_step}; time {format_time(time.time() - time_t0)}")


def evaluate(model, val_dataloader, val_sampler, config, data_step):
    time_t0 = time.time()
    evaluation_loss = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader, desc='Step')):
            val_sampler.set_epoch(step)
            input_ids, attention_mask, token_type_ids, labels = batch[0].long().to(config.device), \
                batch[1].long().to(config.device), batch[2].long().to(config.device), batch[3].long().to(config.device)
            with autocast():
                output = model(input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               output_hidden_states=True,
                               )
                loss = output[0]
            evaluation_loss.append(loss.item())

            if (step+1) % 500 == 0:
                print("Step {} evaluation loss: {}".format(step+1, sum(evaluation_loss)/len(evaluation_loss)))

    eval_loss = sum(evaluation_loss)/len(evaluation_loss)
    time_t1 = time.time()
    cost_time = format_time(time_t1 - time_t0)
    print("Evaluation loss: {} ; Data iteration{} cost time: {}".format(eval_loss, data_step, cost_time))


def main(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        dist.barrier()
        ddp = True
    else:
        ddp = False

    batch_size = args.batch_size
    vocab_size = 5000

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model = buildBERT(vocab_size)
    model.cuda(args.local_rank)
    if ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1

    train_iter = search_path("attention", "./tokenized_data/train/")
    val_iter = search_path("attention", "./tokenized_data/val/")

    optimizer = AdamW(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay)
    scaler = GradScaler()

    for epoch in tqdm(range(args.epochs), desc="Training Epoch"):
        for step in tqdm(range(train_iter), desc="training iteration"):
            input_ids_train = torch.load("./tokenized_data/train/train_input_ids{}.pt".format(step))
            attention_masks_train = torch.load("./tokenized_data/train/train_attention_mask{}.pt".format(step))
            token_type_ids_train = torch.load("./tokenized_data/train/train_token_type_ids{}.pt".format(step))
            labels_train = torch.load("./tokenized_data/train/train_labels{}.pt".format(step))

            input_ids_train = torch.cat(input_ids_train, dim=0)
            attention_masks_train = torch.cat(attention_masks_train, dim=0)
            token_type_ids_train = torch.cat(token_type_ids_train, dim=0)
            labels_train = torch.cat(labels_train, dim=0)

            train_list = [input_ids_train, attention_masks_train, token_type_ids_train, labels_train]
            train_dataloader, train_sampler = generate_dataloader(train_list, "train", ddp, batch_size)

            config = Config()
            config.training_config(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay, device=device)

            train(model, train_dataloader, train_sampler, config, optimizer, scaler, step)

        for step in tqdm(range(val_iter), desc="evaluation iteration"):
            input_ids_val = torch.load("./tokenized_data/val/val_input_ids{}.pt".format(step))
            attention_masks_val = torch.load("./tokenized_data/val/val_attention_mask{}.pt".format(step))
            token_type_ids_val = torch.load("./tokenized_data/val/val_token_type_ids{}.pt".format(step))
            labels_val = torch.load("./tokenized_data/val/val_labels{}.pt".format(step))

            input_ids_val = torch.cat(input_ids_val, dim=0)
            attention_masks_val = torch.cat(attention_masks_val, dim=0)
            token_type_ids_val = torch.cat(token_type_ids_val, dim=0)
            labels_val = torch.cat(labels_val, dim=0)

            val_list = [input_ids_val, attention_masks_val, token_type_ids_val, labels_val]
            val_dataloader, val_sampler = generate_dataloader(val_list, "val", ddp, batch_size)

            config = Config()
            config.training_config(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay, device=device)

            evaluate(model, val_dataloader, val_sampler, config, step)

    torch.save(model, 'bert_model/urlBERT.pt')


if __name__ == "__main__":

    seed_val = 2024
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    args = options.args_parser()
    main(args)
