from dataclasses import dataclass
import torch # the all life torch library
import torch.nn as nn # Neural Network stuff
from torch.nn import functional as F #pydantic
import pydantic
import math

device = 'cuda'
import tiktoken

"""
max_return_sequences = 7 #Number of answers
max_length = 100 #Numbers of characters
enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode("I like dogs because")

print(tokens,type(tokens[0]))

tokens=torch.tensor(tokens,dtype=torch.long)

tokens=tokens.unsqueeze(0).repeat(max_return_sequences,1)

print(tokens)

x=tokens.to('cuda')
"""

with open('/content/input.txt','r',encoding='utf-8') as f:
  text = f.read()
text= text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T) # input to transformer
y = buf[1:].view(B,T) # targets, to calculate loss
x.to('cuda')
y.to('cuda')

# deco dataclass -> Basemodel
# Config = dataclass(Config), then dataclass is a function!

class Config(pydantic.BaseModel):
  n_layer   :int = 12  # number of layer in the transformer arquitecture
  block_size:int = 1024 # block size, which is the context
  vocab_size:int = 50257 #  quantity of words, tokens, 5000 words + 256 + 1  tokenization, so the matrix embedding should be 768x50257,
                       # and what about the unembedding matrix
  n_head    :int = 12  # numbers of head per layer, how we have 12 layers so we have 144 heads.
  n_embd    :int = 768 # number of parameters for each token, one word could be expresed using 768
                        # We can't try with another hyper parameter. We would need more parameters!

class CausalSelfAttention(nn.Module): #Another class, for what reason we use the Causal name?
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # 768/12 = 64 features per head.
    # key, query, value projection for all heads, (each) in batch (tensor)) How you specify the batch size

    self.c_attn = nn.Linear(config.n_embd,3*config.n_embd) #Linear layer, (dimension grow) # the number three because we need kery, key and value (three)

    #output projection

    self.c_proj = nn.Linear(config.n_embd,config.n_embd) #Linear layer (dimension stays the same)

    #regularization (?)
    self.n_head = config.n_head #config?
    self.n_embd = config.n_embd

    self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size)) # Applies causal (mask),
                                                                                                                  # basically creating a lower triangle
  def forward(self,x):

    B,T,C = x.size() # Batch Size=> how prompts are being processing at the same time
                     # T sequence length (contex size) numbers of word in one prompt sentence=>1024 (block size) ,
                     # C=embedding size = n_embd



    qkv = self.c_attn(x) # A linear layer, that obtain the key, query. matrices

    q,k,v = qkv.split(self.n_embd,dim=2) # split into three tensors each (B,T,C)

    k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # A tensor shape (B,T,12,64) and transpose for (B,12,T,64) - sense
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # same
    q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # same

    att = (q@k.transpose(-2,-1))/math.sqrt(k.size(-1)) # context using kery and key, tranpose for a good multiplication matrix result matrix TXT word x word attend
                                                       # ; divides for sqrt (64).
    att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf')) # mask , -infi
    att = F.softmax(att,dim=-1) #Softmax, att turns into a vector of numbers

    y = att @ v # y return to a tensor
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.c_proj(y)

    return y

class MLP(nn.Module): # There's no communication between tokens in this step MULTILAYER PERCEPTRON

  def __init__(self,config):
    super().__init__()
    self.c_fc  = nn.Linear(config.n_embd,4*config.n_embd) #Linear layer that take the matrix to a space with more dimensions
    self.gelu  = nn.GELU(approximate='tanh') # Apply the pseudo activation function (?) than
    self.c_proj = nn.Linear(4*config.n_embd,config.n_embd) #return to the

  def forward(self,x): # practice
    x=self.c_fc(x)
    x=self.gelu(x) # gelu->gaussian error linear function
    x=self.c_proj(x)
    return x #straighforward

class Layer(nn.Module): #This contains the three classes.

  def __init__(self,config): #this config take sense in the initialization of instances
    super().__init__() #inherits, what inherits?

    self.ln_1 = nn.LayerNorm(config.n_embd) # I really don't understand what make this. pytorch: Applies Layer Normalization over a mini-batch of inputs.
    self.attn = CausalSelfAttention(config) #apply attention, (made by hand)
    self.ln_2 = nn.LayerNorm(config.n_embd) #normalize - residual, there exist a paper for understand this, that I wish.
    self.mlp  = MLP(config) # an instance of the MLP class that implements the ffn steps where the tokens are treated like individuals.

  def forward(self,x):# the question what recieves? a batch?

    x += self.attn(self.ln_1(x))
    x += self.mlp(self.ln_2(x))

    return x #So, this is called Pre-LN (normalize before computation).

from transformers import GPT2LMHeadModel # huggin face stuff

class MODEL(nn.Module): #lets make a model=MODEl() #so this is
  def __init__(self,config):

    super().__init__() # that super is weird, for what I want it?

    self.config=config

    self.transformer=nn.ModuleDict(dict(

        wte=nn.Embedding(config.vocab_size,config.n_embd),      # wte=weight tokenization embedding, creating a tensor with that dimension

        wpe=nn.Embedding(config.block_size,config.n_embd),      # wpe= weight possitional encodding, another a tensor basically to count the "context"

        #Layers = [Layer(config) for _ in range(config.n_layer)], # Creating the 12 Layers.

        h=nn.ModuleList([Layer(config) for _ in range(config.n_layer)]), # Module list for

        ln_f=nn.LayerNorm(config.n_embd) # Another normalization!

        )
    ) #Transformer architecture prepare the Embedding, charge the head for each layer and add the Normalization layer, basically makes everything.
    # What return this??

    self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False) #a linear layer with input config.n_emb, output config.vocab_size


  # let's charge parameters from hugging face repo.
  # -------------start---------------
  #  deco classmethod. from_pretrained=classmethod(from_pretrained)
  #  cls, what it means?
  #
  # classmethod
  def from_pretrained(cls, model_type): #well is only "download" a bunch of tensors.
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints

        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints



        config = Config(**config_args)

        model = MODEL(config)

        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

  from_pretrained=classmethod(from_pretrained) #decorator make it clean.

  #---------end---------------

  def forward(self, idx,target=None): #So recievec idx wich is a
    B,T = idx.size() # idx is a matrix, B is the batch size and T is the secuence lengh

    #no matter the embedding dimension? no obvious

    tok_emb = self.transformer.wte(idx) #

    pos_emb = self.transformer.wpe(torch.arange(T,device=idx.device)) #possitional encodding. The same but creating a positional embedding matrix

    x=tok_emb+pos_emb #yes the sum, x contains all the information, of the input

    for block in self.transformer.h: #transformers acting
      x = block(x)
    x = self.transformer.ln_f(x) #normalize it. residual optimize stuff

    logits = self.lm_head(x)
    target = None
    if target is not None:
      loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
    return logits, loss # so MODEL return logits raw number , ready to applied a softmax and then a topk

"""GOAL:Obtain the context of one word
FOR THAT WE HAVE: key, query, value for each token,
"""

model = MODEL(Config()) # Randoms parameters
model.to(device)
#model = MODEL.from_pretrained('gpt2') #creating and charging parameters for the model
#model.eval() #evaluation mode, kill the gradients?

logits, loss = model(x,y)
print(logits,loss)

import sys; sys.exit(0)

import tiktoken                      #to tokenize, I don't made a tokenizer!! this is a completele decoder transformer
enc = tiktoken.get_encoding('gpt2') #tokens specificaly to gpt2!

torch.manual_seed(42) # random seed
torch.cuda.manual_seed(42) # manual_seed
while x.size(1) < max_length:

  with torch.no_grad(): # Let's take the
    logits = model(x)  # first call to MODEL IMPORTANT, shape (1,T,vocab_size) , (1,number of words, features per word)
    logits = logits[:,-1,:] #only last token
    probs = F.softmax(logits,dim=-1) #Softmax
    topk_probs, topk_indices =torch.topk(probs,50,dim=-1) #choose the 50 "words" more probably
    ix = torch.multinomial(topk_probs,1) # from those 50 choose randomly one.
    xcol = torch.gather(topk_indices,-1,ix) # assign that choose?
    x = torch.cat((x,xcol),dim=1) #append to form a sentence

for i in range(max_return_sequences):
  tokens = x[i,:max_length].tolist()
  decoded = enc.decode(tokens)
  print(">",decoded)