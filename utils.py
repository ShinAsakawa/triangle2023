import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import colored
import math
import random
import numpy as np
import time
import jaconv

import operator
import matplotlib.pyplot as plt
import japanize_matplotlib

device="cuda:0" if torch.cuda.is_available() else "cpu"

def psylex_word_char_histgram(
    _dict:dict=None, #psylex71.train_data,
    key:str='orig',  # or 'phon'
    title:str=None,
    topN:int=100,
    figsize=(20,4),
    figsize2=(14,4)):

    if title == None:
        title = key
    chr_count, len_count = {}, {}
    for k, v in _dict.items():
        wrd = v[key]
        wrd_len = len(wrd)
        for ch in wrd:
            if ch in chr_count:
                chr_count[ch] += 1
            else:
                chr_count[ch] = 1

        if wrd_len in len_count:
            len_count[wrd_len] += 1
        else:
            len_count[wrd_len] = 1

    N_chr=np.array([v for v in chr_count.values()]).sum()

    if topN > len(chr_count):
        topN = len(chr_count)

    chr_count_sorted = sorted(chr_count.items(), key=operator.itemgetter(1), reverse=True)
    plt.figure(figsize=figsize)
    plt.bar(range(topN), [x[1]/N_chr for x in chr_count_sorted[:topN]])
    plt.xticks(ticks=range(topN), labels=[c[0] for c in chr_count_sorted[:topN]])

    if topN == len(chr_count):
        plt.title(f'{title}項目頻度')
    else:
        plt.title(f'{title}項目頻度 (上位:{topN} 語)')
    plt.ylabel('相対頻度')
    plt.show()


    N_len=np.array([v for v in len_count.values()]).sum()

    len_count_sorted = sorted(len_count.items(), key=operator.itemgetter(0), reverse=False)
    plt.figure(figsize=figsize2)
    plt.bar(range(len(len_count_sorted)), [x[1]/N_len for x in len_count_sorted])
    plt.xticks(ticks=range(len(len_count_sorted)), labels=[c[0] for c in len_count_sorted])
    plt.ylabel(f'{title}相対頻度')
    plt.title(f'{title}項目長頻度')
    plt.show()



def convert_ids2tensor(
    sentence_ids:list,
    device:torch.device=device):

    """数値 ID リストをテンソルに変換
    例えば，[0,1,2] -> tensor([[0],[1],[2]])
    """
    return torch.tensor(sentence_ids, dtype=torch.long, device=device).view(-1, 1)

import time
import math

def asMinutes(s:int)->str:
    """時間変数を見やすいように，分と秒に変換して返す"""
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m):2d}分 {int(s):2d}秒'


def timeSince(since:time.time,
            percent:time.time)->str:
    """開始時刻 since と，現在の処理が全処理中に示す割合 percent を与えて，経過時間と残り時間を計算して表示する"""
    now = time.time()  #現在時刻を取得
    s = now - since    # 開始時刻から現在までの経過時間を計算
    #s = since - now
    es = s / (percent) # 経過時間を現在までの処理割合で割って終了予想時間を計算
    rs = es - s        # 終了予想時刻から経過した時間を引いて残り時間を計算

    return f'経過時間:{asMinutes(s)} (残り時間 {asMinutes(rs)})'



def calc_accuracy(
    _dataset,
    encoder,
    decoder,
    max_length=None,
    source_vocab=None,
    target_vocab=None,
    source_ids=None,
    target_ids=None,
    isPrint=False):

    ok_count = 0
    for i in range(_dataset.__len__()):
        _input_ids, _target_ids = _dataset.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(
            encoder=encoder,
            decoder=decoder,
            input_ids=_input_ids,
            max_length=max_length,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            source_ids=source_ids,
            target_ids=target_ids,
        )
        ok_count += 1 if _target_ids == _output_ids else 0
        if (_target_ids != _output_ids) and (isPrint):
            print(i, _target_ids == _output_ids, _output_words, _input_ids, _target_ids)

    return ok_count/_dataset.__len__()


def evaluate(
    encoder:torch.nn.Module,
    decoder:torch.nn.Module,
    input_ids:list=None,
    max_length:int=1,
    source_vocab:list=None,
    target_vocab:list=None,
    #source_ids:list=None,
    #target_ids:list=None,
    device:torch.device=device):

    with torch.no_grad():
        input_tensor = convert_ids2tensor(input_ids)
        input_length = input_tensor.size()[0]
        encoder_hid = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.n_hid, device=device)

        for ei in range(input_length):
            encoder_out, encoder_hid = encoder(input_tensor[ei], encoder_hid)
            encoder_outputs[ei] += encoder_out[0, 0]

        decoder_inp = torch.tensor([[source_vocab.index('<SOW>')]], device=device)
        decoder_hid = encoder_hid

        decoded_words, decoded_ids = [], []  # decoded_ids を追加
        decoder_attns = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_out, decoder_hid, decoder_attn = decoder(
                decoder_inp, decoder_hid, encoder_outputs, device=device)
            decoder_attns[di] = decoder_attn.data
            topv, topi = decoder_out.data.topk(1)
            decoded_ids.append(int(topi.squeeze().detach())) # decoded_ids に追加
            if topi.item() == target_vocab.index('<EOW>'):
                decoded_words.append('<EOW>')
                break
            else:
                decoded_words.append(target_vocab[topi.item()])

            decoder_inp = topi.squeeze().detach()

        return decoded_words, decoded_ids, decoder_attns[:di + 1]  # decoded_ids を返すように変更


def check_vals_performance(
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    _dataset:torch.utils.data.Dataset=None,
    max_length:int=0,
    source_vocab:list=None,
    target_vocab:list=None,
    #source_ids=None,
    #target_ids=None,
    device:torch.device=device):

    if _dataset == None or encoder == None or decoder == None or max_length == 0 or source_vocab == None:
        print('Something wrong')
        sys.exit()
    for _x in _dataset:
        ok_count = 0
        #for i in range(_dataset.__len__()):
        for i in range(_dataset[_x].__len__()):
            #_input_ids, _target_ids = _dataset.__getitem__(i)
            _input_ids, _target_ids = _dataset[_x].__getitem__(i)
            _output_words, _output_ids, _attentions = evaluate(
                encoder=encoder, decoder=decoder,
                source_vocab=source_vocab, target_vocab=target_vocab,
                input_ids=_input_ids,
                max_length=max_length,
                #source_ids=source_ids, #target_ids=target_ids,
                device=device)
            ok_count += 1 if _target_ids == _output_ids else 0
        #print(f'{_x}:{ok_count/_dataset.__len__():.3f},',end="")
        print(f'{_x}:{ok_count/_dataset[_x].__len__():.3f},',end=" ")
    print()


def _train(
    input_tensor:torch.Tensor=None,
    target_tensor:torch.Tensor=None,
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    encoder_optimizer:torch.optim=None,
    decoder_optimizer:torch.optim=None,
    criterion:torch.nn.modules.loss=torch.nn.modules.loss.CrossEntropyLoss,
    max_length:int=1,
    target_vocab:list=None,
    teacher_forcing_ratio:float=0.,
    device:torch.device=device)->float:

    """inpute_tensor (torch.Tensor() に変換済の入力系列) を 1 つ受け取って，
    encoder と decoder の訓練を行う
    """

    encoder_hidden = encoder.initHidden() # 符号化器の中間層を初期化
    encoder_optimizer.zero_grad()         # 符号化器の最適化関数の初期化
    decoder_optimizer.zero_grad()         # 復号化器の最適化関数の初期化

    input_length = input_tensor.size(0)   # 0 次元目が系列であることを仮定
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(input_length, encoder.n_hid, device=device)
    #encoder_outputs = torch.zeros(max_length, encoder.n_hid, device=device)

    loss = 0.  # 損失関数値
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            inp=input_tensor[ei],
            hid=encoder_hidden,
            device=device)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[target_vocab.index('<SOW>')]], device=device)
    decoder_hidden = encoder_hidden

    ok_flag = True
    # 教師強制をするか否かを確率的に決める
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing: # 教師強制する場合 Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                        decoder_hidden,
                                                                        encoder_outputs,
                                                                        device=device)
            decoder_input = target_tensor[di]      # 教師強制 する

            loss += criterion(decoder_output, target_tensor[di])
            ok_flag = (ok_flag) and (decoder_output.argmax() == target_tensor[di].detach().cpu().numpy()[0])
            if decoder_input.item() == target_vocab.index('<EOW>'):
                break

    else: # 教師強制しない場合 Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                        decoder_hidden,
                                                                        encoder_outputs,
                                                                        device=device)
            topv, topi = decoder_output.topk(1)     # 教師強制しない
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            ok_flag = (ok_flag) and (decoder_output.argmax() == target_tensor[di].detach().cpu().numpy()[0])
            if decoder_input.item() == target_vocab.index('<EOW>'):
                break

    loss.backward()           # 誤差逆伝播
    encoder_optimizer.step()  # encoder の学習
    decoder_optimizer.step()  # decoder の学習
    return loss.item() / target_length, ok_flag


def _fit(encoder:torch.nn.Module,
         decoder:torch.nn.Module,
         epochs:int=1,
         lr:float=0.0001,
         n_sample:int=3,
         teacher_forcing_ratio=False,
         train_dataset:torch.utils.data.Dataset=None,
         val_dataset:dict=None,
         source_vocab:list=None,
         target_vocab:list=None,
         source_ids:str=None,
         target_ids:list=None,
         params:dict=None,
         max_length:int=1,
         device:torch.device=device,
         #device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )->list:

    start_time = time.time()

    encoder.train()
    decoder.train()
    encoder_optimizer = params['optim_func'](encoder.parameters(), lr=lr)
    decoder_optimizer = params['optim_func'](decoder.parameters(), lr=lr)
    criterion = params['loss_func']
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        ok_count = 0

        #エポックごとに学習順をシャッフルする
        learning_order = np.random.permutation(train_dataset.__len__())

        for i in range(train_dataset.__len__()):
            x = learning_order[i]   # ランダムにデータを取り出す
            input_ids, target_ids = train_dataset.__getitem__(x)
            input_tensor = convert_ids2tensor(input_ids)
            target_tensor = convert_ids2tensor(target_ids)

            #訓練の実施
            loss, ok_flag = _train(input_tensor=input_tensor,
                                   target_tensor=target_tensor,
                                   encoder=encoder,
                                   decoder=decoder,
                                   encoder_optimizer=encoder_optimizer,
                                   decoder_optimizer=decoder_optimizer,
                                   criterion=criterion,
                                   max_length=max_length,
                                   target_vocab=target_vocab,
                                   teacher_forcing_ratio=teacher_forcing_ratio,
                                   device=device)
            epoch_loss += loss
            ok_count += 1 if ok_flag else 0


        losses.append(epoch_loss/train_dataset.__len__())
        print(colored(f'エポック:{epoch:2d} 損失:{epoch_loss/train_dataset.__len__():.2f}', 'blue', attrs=['bold']),
              colored(f'{timeSince(start_time, (epoch+1) * train_dataset.__len__()/(epochs * train_dataset.__len__()))}',
                      'cyan', attrs=['bold']),
              colored(f'訓練データの精度:{ok_count/train_dataset.__len__():.3f}', 'blue', attrs=['bold']))

        check_vals_performance(_dataset=val_dataset,
                               encoder=encoder,
                               decoder=decoder,
                               max_length=max_length,
                               source_vocab=source_vocab,
                               target_vocab=target_vocab,
                               source_ids=source_ids,
                               target_ids=target_ids)
        if n_sample > 0:
            evaluateRandomly(encoder, decoder, n=n_sample)

    return losses
