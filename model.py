import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

device="cuda:0" if torch.cuda.is_available() else "cpu"

# encoder, decoder の再定義
# エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
# この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ

import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """RNNによる符号化器"""
    def __init__(self,
            n_inp:int=0,
            n_hid:int=0):
        super().__init__()
        self.n_hid = n_hid
        self.n_inp = n_inp

        self.embedding = nn.Embedding(num_embeddings=n_inp, embedding_dim=n_hid)
        self.gru = nn.GRU(input_size=n_hid, hidden_size=n_hid)

    def forward(self,
                inp:torch.Tensor=0,
                hid:torch.Tensor=0,
                device=device
               ):
        embedded = self.embedding(inp).view(1, 1, -1)
        out = embedded
        out, hid = self.gru(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


class AttnDecoderRNN(nn.Module):
    """注意付き復号化器の定義"""
    def __init__(self,
                 n_hid:int=0,
                 n_out:int=0,
                 dropout_p:float=0.0,
                 max_length:int=0):
        super().__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.dropout_p = dropout_p
        self.max_length = max_length

        # n_out と n_inp は同じ特徴数であるから，n_out が 入力特徴数となる
        self.embedding = nn.Embedding(num_embeddings=n_out, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)

        self.dropout = nn.Dropout(p=dropout_p)

        # GRU には batch_first オプションをつけていない。これは，一系列ずつしかしょりしないため
        self.gru = nn.GRU(input_size=n_hid, hidden_size=n_hid)

        # 最終出力
        self.out_layer = nn.Linear(in_features=n_hid, out_features=n_out)

    def forward(self,
                inp:torch.Tensor=None,  # 旧版では int だが正しくは torch.Tensor
                hid:torch.Tensor=None,  # 旧版では int だが正しくは torch.Tensor
                encoder_outputs:torch.Tensor=None,
                device:torch.device=device):
        embedded = self.embedding(inp).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 注意の重みを計算
        # 入力 (embedded[0]) と 中間層 (hidden[0]) とを第 2 次元に沿って連結 (torch.cat(dim=1))
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hid[0]), 1)), dim=1)

        # エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
        # この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ
        #print(f'attn_weights.unsqueeze(0).size():{attn_weights.unsqueeze(0).size()}',
        #      f'encoder_outputs.unsqueeze(0).size():{encoder_outputs.unsqueeze(0).size()}'
        #)
        #sys.exit()
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        out = torch.cat((embedded[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)

        out = F.relu(out)
        out, hid = self.gru(out, hid)

        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


# encoder = EncoderRNN(
#     n_inp=len(_vocab.source_vocab),
#     n_hid=params['hidden_size']).to(device)

# decoder = AttnDecoderRNN(
#     n_hid=params['hidden_size'],
#     n_out=len(_vocab.target_vocab),
#     dropout_p=params['dropout_p'],
#     max_length=_vocab.source_maxlen).to(device)


############################################################################

# class EncoderRNN(nn.Module):
#     """RNNによる符号化器"""
#     def __init__(self,
#             n_inp:int=0,
#             n_hid:int=0):
#             #device=device):
#         super().__init__()
#         self.n_hid = n_hid if n_hid != 0 else 8
#         self.n_inp = n_inp if n_inp != 0 else 8

#         self.embedding = nn.Embedding(n_inp, n_hid)
#         self.gru = nn.GRU(n_hid, n_hid)

#     def forward(self,
#                 inp:int=0,
#                 hid:int=0,
#                 device=device
#                ):
#         embedded = self.embedding(inp).view(1, 1, -1)
#         out = embedded
#         out, hid = self.gru(out, hid)
#         return out, hid

#     def initHidden(self)->torch.Tensor:
#         return torch.zeros(1, 1, self.n_hid, device=device)


# class AttnDecoderRNN(nn.Module):
#     """注意付き復号化器の定義"""
#     def __init__(self,
#                  n_hid:int=0,
#                  n_out:int=0,
#                  dropout_p:float=0.0,
#                  max_length:int=0):
#         super().__init__()
#         self.n_hid = n_hid
#         self.n_out = n_out
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.n_out, self.n_hid)
#         self.attn = nn.Linear(self.n_hid * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.n_hid * 2, self.n_hid)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.n_hid, self.n_hid)
#         self.out = nn.Linear(self.n_hid, self.n_out)

#     def forward(self,
#                 inp:int=0,
#                 hid:int=0,
#                 encoder_outputs:torch.Tensor=None,
#                 device=device):
#         embedded = self.embedding(inp).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hid[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

#         out = torch.cat((embedded[0], attn_applied[0]), 1)
#         out = self.attn_combine(out).unsqueeze(0)

#         out = F.relu(out)
#         out, hid = self.gru(out, hid)

#         out = F.log_softmax(self.out(out[0]), dim=1)
#         return out, hid, attn_weights

#     def initHidden(self)->torch.Tensor:
#         return torch.zeros(1, 1, self.n_hid, device=device)

