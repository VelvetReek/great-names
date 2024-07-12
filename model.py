import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyperparameters
block_size = 8
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of hidden units per layer
max_steps = 20000
batch_size = 32
lr = 1e-2

# read in all the words
words = open('names.txt', 'r').read().splitlines()
words = [word.lower() for word in words] # convert to lower case
random.seed(128847)
random.shuffle(words) # shuffle the words

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0 
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# Preapre the dataset for training, validation and testing
def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  # print(X.shape, Y.shape)
  return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

# --------------------------------------------------------------------------------------------------------------------------
# Creating classes equivalent to nn.Module for indepth understanding

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
  
class Embedding:

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim)) 
  
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]
  
class Flaten:
  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out
  
  def parameters(self):
    return []
  
# --------------------------------------------------------------------------------------------------------------------------

# Define the model
class NameGen():
  
  def __init__(self):
    super(NameGen, self).__init__()
    self.embedding = Embedding(vocab_size, n_embd)
    self.linear1 = Linear(n_embd*block_size, n_hidden)
    self.bn1 = BatchNorm1d(n_hidden)
    self.tanh = Tanh()
    self.linear2 = Linear(n_hidden, vocab_size)
  
  def __call__(self, x, targets=None):
    x = self.embedding(x)
    x = x.view(x.shape[0], -1)
    x = self.linear1(x)
    x = self.bn1(x)
    x = self.tanh(x)
    x = self.linear2(x)
    if targets is None:
      loss = None
    else:
      loss = F.cross_entropy(x, targets)
    return x, loss
    
  
  def parameters(self):
    return self.embedding.parameters() + self.linear1.parameters() + self.bn1.parameters() + self.linear2.parameters()
  
  def train(self):
    self.training = True
    self.bn1.training = True

  def eval(self):
    self.training = False
    self.bn1.training = False
  
  def generate(self, max_len = 15):
    self.eval()
    context = [0] * block_size
    out = []
    for i in range(max_len):
      logits, _ = self(torch.tensor([context]))
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs[0], 1).item()
      if ix == 0: break
      out.append(ix)
      context = context[1:] + [ix]

      
    return ''.join(itos[ix] for ix in out)
  
model = NameGen()

# This is to make initial weights of last layer less confident
# model.layers[-1].weight /= 10

# Parameters intialization
parameters = model.parameters()
print(f'{sum(p.numel() for p in parameters)} parameters in total')
for p in parameters:
  p.requires_grad = True

# --------------------------------------------------------------------------------------------------------------------------
# Setting Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
losses = []

# Training the model
for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # evaluate the loss
  logits, loss = model(Xb, Yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  losses.append(loss.log10().item())
  # break

# Plot the loss during training
plt.plot(torch.tensor(losses).view(-1, 1000).mean(1))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# --------------------------------------------------------------------------------------------------------------------------
# evaluate the loss
model.eval()
@torch.no_grad()
def split_loss(split):
    x,y ={
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    _, loss = model(x, y)
    print(f'{split} loss: {loss.item():.4f}')

split_loss('train')
split_loss('val')

# --------------------------------------------------------------------------------------------------------------------------
# sample from the model
print('Samples from the model:\n')
for _ in range(10):
    print(model.generate(50))