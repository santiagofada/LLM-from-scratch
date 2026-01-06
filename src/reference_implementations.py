import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


################
## Chapter 02 ##
################
class GPTDatasetV1(Dataset):
    # Builds (input_ids, target_ids) pairs for next-token prediction.
    # Each sample is a fixed-length window of tokens, and the target is the same window shifted by 1.
    def __init__(self, txt, tokenizer, max_length, stride):
        # txt: raw text
        # tokenizer: gpt2 encoder
        # max_length: seq length of an example in training
        # stride: sliding window step
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # `stride` controls the overlap between consecutive windows (smaller stride => more overlap => more samples).
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # txt: raw text
    # batch_size: batch size
    # max_length: seq length of an example in training
    # stride: window step
    # shuffle: shuffle batches
    # drop_last: drop remainder
    # num_workers: worker processes

    # Returns batches of (input_ids, target_ids) tensors.

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader


################
## Chapter 03 ##
################
class MultiHeadAttention(nn.Module):
    # Multi-head self-attention with a causal (look-ahead) mask.
    # The mask prevents each token from attending to future tokens, enabling autoregressive decoding.
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # d_in: input dim (emb)
        # d_out: output dim (model)
        # context_length: max context in inference
        # dropout: dropout prob
        # num_heads: parallel heads
        # qkv_bias: qkv bias
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            # Upper-triangular mask (above diagonal) marks positions we must block (future tokens).
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
        # this will result in errors in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # Each head works on a smaller subspace of size `head_dim`.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # We set masked (future) positions to -inf so their softmax probability becomes 0.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim) for numerical stability (standard scaled dot-product attention).
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


################
## Chapter 04 ##
################

class LayerNorm(nn.Module):
    # Implementa una normalización de capa (LayerNorm) simple sobre la última dimensión.
    # Es una implementación personalizada similar a nn.LayerNorm de PyTorch.
    def __init__(self, emb_dim):
        # emb_dim: dimensión del embedding a normalizar
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Normaliza cada embedding de token de forma independiente.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # Aplica una escala y un desplazamiento aprendibles.
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    # Implementa la función de activación GELU (Gaussian Error Linear Unit).
    # Utiliza la aproximación con `tanh`, común en modelos de estilo GPT.
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    # Define una red feed-forward (MLP) que se aplica a cada posición.
    # Consiste en: expandir -> no linealidad -> proyectar de nuevo.
    def __init__(self, cfg):
        # cfg: diccionario de configuración del modelo.
        super().__init__()
        self.layers = nn.Sequential(
            # Expande la dimensión del embedding por un factor de 4.
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            # Proyecta de nuevo a la dimensión original del embedding.
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    # Bloque Transformer con pre-normalización y conexiones residuales.
    def __init__(self, cfg):
        # cfg: diccionario de configuración del modelo.
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # x: embeddings de tokens (B, T, D)

        # Conexión residual (atajo) para el bloque de atención.
        shortcut = x
        # Pre-normalización: se aplica LayerNorm antes de la atención.
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Se suma la entrada original (conexión residual).

        # Conexión residual para el bloque feed-forward.
        shortcut = x
        # Pre-normalización: se aplica LayerNorm antes del feed-forward.
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Se suma la entrada original.

        return x


class GPTModel(nn.Module):
    # Modelo GPT minimalista: embeddings de token+posición -> bloques Transformer -> cabeza de lenguaje.
    def __init__(self, cfg):
        # cfg: diccionario de configuración del modelo.
        # cfg["vocab_size"]: tamaño del vocabulario del tokenizador.
        # cfg["emb_dim"]: dimensión del embedding (D).
        # cfg["context_length"]: contexto máximo en inferencia.
        # cfg["drop_rate"]: probabilidad de dropout.
        # cfg["n_layers"]: número de bloques Transformer.
        super().__init__()
        # Capa de embedding para los tokens del vocabulario.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Capa de embedding para las posiciones de los tokens.
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Pila de bloques Transformer.
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Normalización final antes de la capa de salida.
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Capa de salida (cabeza de lenguaje) que proyecta al tamaño del vocabulario.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx: IDs de los tokens de entrada (B, T)
        batch_size, seq_len = in_idx.shape
        # Obtiene los embeddings de los tokens.
        tok_embeds = self.tok_emb(in_idx)
        # Las posiciones son [0..seq_len-1]; los embeddings posicionales se comparten en el batch.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # Suma los embeddings de token y de posición.
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # Pasa la entrada a través de todos los bloques Transformer.
        x = self.trf_blocks(x)
        # Aplica la normalización final.
        x = self.final_norm(x)
        # Proyecta al espacio del vocabulario para obtener los logits.
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Genera texto usando decodificación "greedy": elige repetidamente el token con el logit más alto.
    # model: modelo de lenguaje que devuelve logits.
    # idx: IDs de los tokens en el contexto actual (B, T).
    # max_new_tokens: número de nuevos tokens a generar.
    # context_size: tamaño de la ventana de contexto que el modelo puede manejar.

    for _ in range(max_new_tokens):
        # Recorta el contexto si excede el tamaño soportado por el modelo.
        # Solo se usan los últimos `context_size` tokens.
        idx_cond = idx[:, -context_size:]

        # Obtiene las predicciones del modelo.
        with torch.no_grad():
            logits = model(idx_cond)

        # Se enfoca solo en el último paso de tiempo para predecir el siguiente token.
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # Obtiene el índice del token con el logit más alto (decodificación greedy).
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Añade el índice muestreado a la secuencia en ejecución.
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx