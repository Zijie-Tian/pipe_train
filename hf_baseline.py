import os
# Important: the following env vars is a hack to emulated distributed environment on a single GPU. Remove all of the env vars if you run 
# with more than one GPU and the torch.distributed or the deepspeed launcher
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9994'
# os.environ['RANK'] = "0"
# os.environ['LOCAL_RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
from torchtext.datasets import WikiText2

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=2, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./ds_config_gpt_j.json')
model = AutoModelForCausalLM.from_pretrained("gpt-j-6B").cpu()
model.resize_token_embeddings(len(tokenizer))
# descriptions = pd.read_csv('netflix_titles.csv')['description']
# max_length = max([len(tokenizer.encode(description)) for description in descriptions])
# print("Max length: {}".format(max_length))

train_iter, val_iter, test_iter = WikiText2()
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) 
def encoding_data(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i):
    bptt = 512
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].view(-1)
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data, target

class WT2_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, bsz):
        # self.data = batchify(encoding_data(data), 1024)
        # Here the data is a sequence of encoded text.
        # Pretend to have a batch dimension.
        _train_data = encoding_data(data)
        train_data = batchify(_train_data, bsz)

        self.inputs = []
        self.labels = []
        self.nbatch = 600
        for i in range(self.nbatch):
            input_ids, tgt = get_batch(train_data, i)
            self.inputs.append(input_ids)
            self.labels.append(tgt)

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

# dataset = NetflixDataset(descriptions, tokenizer, max_length=max_length)
# train_size = int(0.9 * len(dataset))
# train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

wt2 = WT2_Dataset2(train_iter, 512)
train_dataset = wt2[:int(len(wt2) * 0.9)]
val_dataset = wt2[int(len(wt2) * 0.9):]

Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[1] for f in data])}).train()
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
