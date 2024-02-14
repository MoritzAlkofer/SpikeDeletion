
from torch.nn import BCELoss
from pytorch_lightning import LightningModule
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
import torch
import math

import sys
sys.path.append('/home/moritz/Desktop/programming/epilepsy_project/librarys')

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,target,pred):
        return torch.sqrt(self.mse(target,pred))
    
# create a regression head for the datset
class RegressionHead(nn.Sequential):
    def __init__(self, emb_size,dropout=0.3):
        super().__init__()
        self.reghead = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.reghead(x)
        return out

# copied from chaoqi
class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size, n_freq):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, 1, freq, time)
        out: (batch, time, emb_size)
        """
        b, _, _, _ = x.shape
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.projection(x)
        return x

# copied from chaoqi
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# copied from chaoqi
class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=int,
        heads=int,
        depth=int,
        n_channels=int,
        n_fft=int,
        hop_length=int,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        signal = []
        for s in range(sample.shape[1]):
            spectral = torch.stft(
                sample[:, s, :],
                n_fft=self.n_fft,  # token length
                hop_length=self.hop_length,  # overlaps
                normalized=False,
                center=False,
                onesided=True,
                return_complex=True            
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb (only for unsupervised pretraining)
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # mean pooling (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb
 
# create a specialized finetuning module
class EEGTransformer(LightningModule):
    def __init__(self,lr,head_dropout,emb_size,weight_decay, heads, depth,n_channels=int,n_fft=int,hop_length=int, emb_mode = False):
        super().__init__()
        self.lr = lr
        self.weight_decay=weight_decay
        self.emb_mode = emb_mode
        # create the BIOT encoder
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth,n_channels=n_channels,n_fft=n_fft,hop_length = hop_length)
        # create the regression head
        self.head = RegressionHead(emb_size,head_dropout)
        self.RMSE = RMSELoss()
        self.loss = BCELoss()

    def forward(self, x):
        x = self.biot(x)
        x = self.head(x)
        return x

    def training_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('train_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('train_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('val_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        if not self.emb_mode:
            signals, labels = batch
            # flatten label
            labels = labels.view(-1, 1).float()
            # generate predictions
            preds = self.forward(signals)
            # compute and log loss
            return preds
        elif self.emb_mode:
            signals, labels = batch
            emb = self.biot(signals)
            return emb

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)

        return optimizer
    