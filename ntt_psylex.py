# NTT psylex71.txt に基づく dataset

import torch
import os
import re
import sys
import numpy as np
import gzip
import jaconv

from termcolor import colored
from tqdm.notebook import tqdm  # jupyter で実行時


class psylex71_dataset(torch.utils.data.Dataset):
    '''
    訓練データとしては，NTT 日本語語彙特性 (天野，近藤, 1999, 三省堂) の頻度データ，実際のファイル名としては `pslex71.txt` から頻度データを読み込んで，高頻度語を訓練データとする。
    ただし，検証データに含まれる単語は訓練データとして用いない。

    検証データとして，以下のいずれかを考える
    1. TLPA (藤田 他, 2000, 「失語症語彙検査」の開発，音声言語医学 42, 179-202)
    2. SALA 上智大学失語症語彙検査

    このオブジェクトクラスでは，
    `phon_vocab`, `orth_vocab`, `ntt_freq`, に加えて，単語の読みについて ntt_orth2hira によって読みを得ることにした。

    * `train_data`, `test_data` という辞書が本体である。
    各辞書の項目には，さらに
    `Vocab_ja.test_data[0].keys() = dict_keys(['orig', 'orth', 'phon', 'orth_ids', 'phon_ids', 'semem'])`

    各モダリティ共通トークンとして以下を設定した
    * <PAD>: 埋め草トークン
    * <EQW>: 単語終端トークン
    * <SOW>: 単語始端トークン
    * <UNK>: 未定義トークン

    このクラスで定義されるデータは 2 つの辞書である。すなわち 1. train_data, 2. tlpa_data である。
    各辞書は，次のような辞書項目を持つ。
    ```
    {0: {'orig': 'バス',
    'yomi': 'ばす',
    'orth': ['バ', 'ス'],
    'orth_ids': [695, 514],
    'orth_r': ['ス', 'バ'],
    'orth_ids_r': ['ス', 'バ'],
    'phon': ['b', 'a', 's', 'u'],
    'phon_ids': [23, 7, 19, 12],
    'phon_r': ['u', 's', 'a', 'b'],
    'phon_ids_r': [12, 19, 7, 23],
    'mora': ['ば', 'す'],
    'mora_r': ['す', 'ば'],
    'mora_ids': [87, 47],
    'mora_p': ['b', 'a', 's', 'u'],
    'mora_p_r': ['s', 'u', 'b', 'a'],
    'mora_p_ids': [6, 5, 31, 35],
    'mora_p_ids_r': [31, 35, 6, 5]},
    ```
    '''

    def __init__(self,
                 traindata_size=10000,     # デフォルト語彙数
                 w2v=None,				   # word2vec (gensim)
                 yomi=None,			       # MeCab を用いた `読み` の取得のため`
                 ps71_fname: str = None,   # NTT 日本語語彙特性の頻度データファイル名
                 stop_list: list = [],	   # ストップ単語リスト：訓練データから排除する単語リスト
                 source: str = 'orth',
                 target: str = 'phon',
                 is_Print: bool = False,
                 ):

        super().__init__()

        if yomi != None:
            self.yomi = yomi
        else:
            #from mecab_settings import yomi
            from ccap.mecab_settings import yomi
            self.yomi = yomi

        # 訓練語彙数の上限 `training_size` を設定
        self.traindata_size = traindata_size

        self.source = source
        self.target = target

        # `self.moraWakachi()` で用いる正規表現のあつまり 各条件を正規表現で表す
        self.c1 = '[うくすつぬふむゆるぐずづぶぷゔ][ぁぃぇぉ]'  # ウ段＋「ァ/ィ/ェ/ォ」
        self.c2 = '[いきしちにひみりぎじぢびぴ][ゃゅぇょ]'  # イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        self.c3 = '[てで][ぃゅ]'  # 「テ/デ」＋「ャ/ィ/ュ/ョ」
        self.c4 = '[ぁ-ゔー]'  # カタカナ１文字（長音含む）
        self.c5 = '[ふ][ゅ]'
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        ##cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
        self.cond = '('+self.c5+'|'+self.c1+'|' + \
            self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        # 以上 `self.moraWakachi()` で用いる正規表現の定義

        self.orth_vocab, self.orth_freq = [
            '<PAD>', '<EOW>', '<SOW>', '<UNK>'], {}
        self.phon_vocab, self.phone_freq = [
            '<PAD>', '<EOW>', '<SOW>', '<UNK>'], {}
        self.phon_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
                           'N', 'a', 'a:', 'e', 'e:', 'i', 'i:', 'i::', 'o', 'o:', 'o::', 'u', 'u:',
                           'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy', 'j', 'k', 'ky',
                           'm', 'my', 'n', 'ny', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z']

        self.mora_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
                           'ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ',
                           'カ', 'ガ', 'キ', 'ギ', 'ク', 'グ', 'ケ', 'ゲ', 'コ', 'ゴ',
                           'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ',
                           'タ', 'ダ', 'チ', 'ヂ', 'ッ', 'ツ', 'ヅ', 'テ', 'デ', 'ト', 'ド',
                           'ナ', 'ニ', 'ヌ', 'ネ', 'ノ',
                           'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ',
                           'マ', 'ミ', 'ム', 'メ', 'モ',
                           'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ',
                           'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ン', 'ー']

        # 全モーラリストを `mora_vocab` として登録
        self.mora_vocab = [
            '<PAD>', '<EOW>', '<SOW>', '<UNK>',
            'ぁ', 'あ', 'ぃ', 'い', 'ぅ', 'う', 'うぃ', 'うぇ', 'うぉ', 'ぇ', 'え', 'お',
            'か', 'が', 'き', 'きゃ', 'きゅ', 'きょ', 'ぎ', 'ぎゃ', 'ぎゅ', 'ぎょ', 'く', 'くぁ', 'くぉ', 'ぐ', 'ぐぁ', 'け', 'げ', 'こ', 'ご',
            'さ', 'ざ', 'し', 'しぇ', 'しゃ', 'しゅ', 'しょ', 'じ', 'じぇ', 'じゃ', 'じゅ', 'じょ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ',
            'た', 'だ', 'ち', 'ちぇ', 'ちゃ', 'ちゅ', 'ちょ', 'ぢ', 'ぢゃ', 'ぢょ', 'っ', 'つ', 'つぁ', 'つぃ', 'つぇ', 'つぉ', 'づ', 'て',
            'てぃ', 'で', 'でぃ', 'でゅ', 'と', 'ど',
            'な', 'に', 'にぇ', 'にゃ', 'にゅ', 'にょ', 'ぬ', 'ね', 'の',
            'は', 'ば', 'ぱ', 'ひ', 'ひゃ', 'ひゅ', 'ひょ', 'び', 'びゃ', 'びゅ', 'びょ', 'ぴ', 'ぴゃ', 'ぴゅ', 'ぴょ',
            'ふ', 'ふぁ', 'ふぃ', 'ふぇ', 'ふぉ', 'ふゅ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ',
            'ま', 'み', 'みゃ', 'みゅ', 'みょ', 'む', 'め', 'も',
            'や', 'ゆ', 'よ', 'ら', 'り', 'りゃ', 'りゅ', 'りょ', 'る', 'れ', 'ろ', 'ゎ', 'わ', 'ゐ', 'ゑ', 'を', 'ん', 'ー',
            # 2022_1017 added
            'ずぃ', 'ぶぇ', 'ぶぃ', 'ぶぁ', 'ゅ', 'ぶぉ', 'いぇ', 'ぉ', 'くぃ', 'ひぇ', 'くぇ', 'ぢゅ', 'りぇ',
        ]

        # モーラに用いる音を表すリストを `mora_p_vocab` として登録
        self.mora_p_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
                             'N', 'a', 'b', 'by', 'ch', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
                             'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'w', 'y', 'z']

        # 母音を表す音から ひらがな への変換表を表す辞書を `vow2hira` として登録
        self.vow2hira = {'a': 'あ', 'i': 'い',
                         'u': 'う', 'e': 'え', 'o': 'お', 'N': 'ん'}

        self.mora_freq = {'<PAD>': 0, '<EOW>': 0, '<SOW>': 0, '<UNK>': 0}
        self.mora_p = {}

        # NTT 日本語語彙特性データから，`self.train_data` を作成
        self.ntt_freq, self.ntt_orth2hira = self.make_ntt_freq_data(
            ps71_fname=ps71_fname)
        self.ntt_freq_vocab = self.set_train_vocab()
        self.train_data, self.excluded_data = {}, []
        max_orth_length, max_phon_length, max_mora_length, max_mora_p_length = 0, 0, 0, 0
        self.train_vocab = []

        num = '０１２３４５６７８９'
        alpha = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'   # ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        hira = 'あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをん'
        kata = 'アイウエオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモヤユヨラリルレロワヰヱヲン'
        onechars = hira+alpha+num  # +kata
        for i, orth in enumerate(onechars):

            if not orth in self.train_vocab:
                self.train_vocab.append(orth)
            try:
                _yomi = self.yomi(orth).strip()
            except:
                print(f'Error: i:{i} orth:{orth}')
                sys.exit()
            hira = jaconv.kata2hira(_yomi)
            phon_juli = jaconv.hiragana2julius(hira).split(' ')

            # 書記素 ID リスト `orth_ids` に書記素を登録
            for o in orth:
                if not o in self.orth_vocab:
                    self.orth_vocab.append(o)
            orth_ids = [self.orth_vocab.index(o) for o in orth]
            phon_ids = [self.phon_vocab.index(
                p) if p in self.phon_vocab else self.phon_vocab.index('<UNK>') for p in phon_juli]

            self.train_data[i] = {
                'orig': orth,
                'orth': orth,
                'yomi': _yomi,
                'phon': phon_juli,
                'phon_ids': phon_ids,
                'orth_ids': orth_ids
            }

        for orth in tqdm(self.ntt_freq_vocab):
            if orth in stop_list:	   # stop list に登録されていたらスキップ
                continue

            if orth in self.train_vocab:  # すでに登録されている単語であればスキップ
                continue
            else:
                self.train_vocab.append(orth)

            n_i = len(self.train_data)

            # 書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
            # 書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る
            _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = self.get7lists_from_orth(
                orth_wrd=orth)

            # 音韻語彙リスト `self.phon_vocab` に音韻が存在していれば True そうでなければ False というリストを作成し，
            # そのリスト無いに False があれば，排除リスト `self.excluded_data` に登録する
            # if False in [True if p in self.phon_vocab else False for p in _phon]:
            #	self.excluded_data.append(orth)
            #	continue

            phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = self.get6ids(
                _phon, _orth, _yomi)
            _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = self.yomi2mora_transform(
                _yomi)
            self.train_data[n_i] = {'orig': orth, 'yomi': _yomi,
                                    'orth': _orth, 'orth_ids': orth_ids, 'orth_r': _orth_r, 'orth_ids_r': orth_ids_r,
                                    'phon': _phon, 'phon_ids': phon_ids, 'phon_r': _phon_r, 'phon_ids_r': phon_ids_r,
                                    'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p,
                                    'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                                    }
            len_orth, len_phon, len_mora, len_mora_p = len(
                _orth), len(_phon), len(_mora), len(_mora_p)
            max_orth_length = len_orth if len_orth > max_orth_length else max_orth_length
            max_phon_length = len_phon if len_phon > max_phon_length else max_phon_length
            max_mora_length = len_mora if len_mora > max_mora_length else max_mora_length
            max_mora_p_length = len_mora_p if len_mora_p > max_mora_p_length else max_mora_p_length

            if len(self.train_data) >= self.traindata_size:  # 上限値に達したら終了する
                #self.train_vocab = [self.train_data[x]['orig'] for x in self.train_data.keys()]
                break

        self.max_orth_length = max_orth_length
        self.max_phon_length = max_phon_length
        self.max_mora_length = max_mora_length
        self.max_mora_p_length = max_mora_p_length

        self.word_list = [v['orig'] for k, v in self.train_data.items()]
        self.order = {i: self.train_data[x]
                      for i, x in enumerate(self.train_data)}
        self.set_source_and_target_from_params(
            source=source, target=target, is_print=is_Print)

    def yomi2mora_transform(self, yomi):
        """ひらがな表記された引数 `yomi` から，日本語の 拍(モーラ)  関係のデータを作成する
        引数:
        yomi:str ひらがな表記された単語 UTF-8 で符号化されていることを仮定している

        戻り値:
        yomi:str 入力された引数
        _mora1:list[str] `_mora` に含まれる長音 `ー` を直前の母音で置き換えた，モーラ単位の分かち書きされた文字列のリスト
        _mora1_r:list[str] `_mora1` を反転させた文字列リスト
        _mora:list[str] `self.moraWakatchi()` によってモーラ単位で分かち書きされた文字列のリスト
        _mora_ids:list[int] `_mora` を対応するモーラ ID で置き換えた整数値からなるリスト
        _mora_p:list[str] `_mora` を silius によって音に変換した文字列リスト
        _mora_p_r:list[str] `_mora_p` の反転リスト
        _mora_p_ids:list[int] `mora_p` の各要素を対応する 音 ID に変換した数値からなるリスト
        _mora_p_ids_r:list[int] `mora_p_ids` の各音を反転させた数値からなるリスト
        _juls:list[str]: `yomi` を julius 変換した音素からなるリスト
        """
        _mora = self.moraWakachi(yomi)  # 一旦モーラ単位の分かち書きを実行して `_mora` に格納

        # 単語をモーラ反転した場合に長音「ー」の音が問題となるので，長音「ー」を母音で置き換えるためのプレースホルダとして. `_mora` を用いる
        _mora1 = _mora.copy()

        # その他のプレースホルダの初期化，モーラ，モーラ毎 ID, モーラ音素，モーラの音素の ID， モーラ音素の反転，モーラ音素の反転 ID リスト
        mora_ids, mora_p, mora_p_ids, mora_p_r, _mora_p_ids_r = [], [], [], [], []
        _m0 = 'ー'  # 長音記号

        for i, _m in enumerate(_mora):  # 各モーラ単位の処理と登録

            __m = _m0 if _m == 'ー' else _m			   # 長音だったら，前音の母音を __m とし，それ以外は自分自身を __m に代入
            _mora1[i] = __m							   # 長音を変換した結果を格納
            mora_ids.append(self.mora_vocab.index(__m))  # モーラを ID 番号に変換
            mora_p += jaconv.hiragana2julius(__m).split()
            # _mora_p += self.mora2jul[__m]				 # モーラを音素に変換して `_mora_p` に格納

            # 変換した音素を音素 ID に変換して，`_mora_p_ids` に格納
            # for _p in jaconv.hiragana2julius(_m).split():
            #	idx = self.phon_vocab.index(_p)
            #	mora_p_ids.append(idx)
            #mora_p_ids = [self.phon_vocab.index(_p) for _p in jaconv.hiragana2julius(__m).split()]
            #_mora_p_ids += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[__m]]

            if not _m in self.mora_freq:  # モーラの頻度表を集計
                self.mora_freq[__m] = 1
            else:
                self.mora_freq[__m] += 1

            if self.hira2julius(__m)[-1] in self.vow2hira:	  # 直前のモーラの最終音素が母音であれば
                # 直前の母音を代入しておく。この処理が 2022_0311 でのポイントであった
                _m0 = self.vow2hira[self.hira2julius(__m)[-1]]

        mora_p_ids = [self.phon_vocab.index(_p) for _p in mora_p]

        # モーラ分かち書きした単語 _mora1 の反転を作成し `_mora1_r` に格納
        _mora1_r = [m for m in _mora1[::-1]]
        mora_p_r = []
        for _m in _mora1_r:				   # 反転した各モーラについて
            # モーラ単位で julius 変換して音素とし `_mora_p_r` に格納
            for _jul in jaconv.hiragana2julius(_m).split():
                mora_p_r.append(_jul)
            #_mora_p_r += self.mora2jul[_m]

            # mora_p_r に格納した音素を音素 ID に変換し mora_p_ids に格納
            #mora_p_ids += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[_m]]

        mora_p_ids_r = [self.phon_vocab.index(_m) for _m in mora_p_r]
        _juls = self.hira2julius(yomi)

        return yomi, _mora1, _mora1_r, _mora, mora_ids, mora_p, mora_p_r, mora_p_ids, mora_p_ids_r, _juls

    def orth2orth_ids(self,
                      orth: str):
        orth_ids = [self.orth_vocab.index(
            ch) if ch in self.orth_vocab else self.orth_vocab.index('<UNK>') for ch in orth]
        return orth_ids

    def phon2phon_ids(self,
                      phon: list):
        phon_ids = [self.phon_vocab.index(
            ph) if ph in self.phon_vocab else self.phon_vocab.index('<UNK>') for ph in phon]
        return phon_ids

    def yomi2phon_ids(self,
                      yomi: str):
        phon_ids = []
        for _jul in self.hira2julius(yomi):
            if _jul in self.phon_vocab:
                ph = self.phon_vocab.index(_jul)
            else:
                ph = self.phon_vocab.index('<UNK>')
            phon_ids.append(ph)
        return phon_ids

    def orth_ids2tkn(self, ids: list):
        return [self.orth_vocab[idx] for idx in ids]

    def orth_tkn2ids(self, tkn: list):
        return [self.orth_vocab.index(_tkn) if _tkn in self.orth_vocab else self.orth_vocab.index('<UNK>') for _tkn in tkn]

    def mora_p_ids2tkn(self, ids: list):
        return [self.mora_p_vocab[idx] for idx in ids]

    def mora_p_tkn2ids(self, tkn: list):
        return [self.mora_p_vocab.index(_tkn) if _tkn in self.mora_p_vocab else self.mora_p_vocab('<UNK>') for _tkn in tkn]

    def mora_ids2tkn(self, ids: list):
        return [self.mora_vocab[idx] for idx in ids]

    def mora_tkn2ids(self, tkn: list):
        return [self.mora_vocab.index(_tkn) if _tkn in self.mora_vocab else self.mora_vocab('<UNK>') for _tkn in tkn]

    def phon_ids2tkn(self, ids: list):
        return [self.phon_vocab[idx] for idx in ids]

    def phon_tkn2ids(self, tkn: list):
        return [self.phon_vocab.index(_tkn) if _tkn in self.phon_vocab else self.phon_vocab.index('<UNK>') for _tkn in tkn]

    def get6ids(self, _phon, _orth, yomi):

        # 音韻 ID リスト `phon_ids` に音素を登録する
        phon_ids = [self.phon_vocab.index(
            p) if p in self.phon_vocab else self.phon_vocab.index('<UNK>') for p in _phon]

        # 直上の音韻 ID リストの逆転を作成
        phon_ids_r = [p_id for p_id in phon_ids[::-1]]

        # 書記素 ID リスト `orth_ids` に書記素を登録
        for o in _orth:
            if not o in self.orth_vocab:
                self.orth_vocab.append(o)
        orth_ids = [self.orth_vocab.index(o) for o in _orth]

        # 直上の書記素 ID リストの逆転を作成
        orth_ids_r = [o_id for o_id in orth_ids[::-1]]
        #orth_ids_r = [o_id for o_id in _orth[::-1]]

        mora_ids = []
        for _p in self.hira2julius(yomi):
            mora_ids.append(self.phon_vocab.index(
                _p) if _p in self.phon_vocab else self.phon_vocab.index('<UNK>'))

        mora_ids_r = [m_id for m_id in mora_ids]

        return phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r

    def moraWakachi(self, hira_text):
        """ ひらがなをモーラ単位で分かち書きする
        https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12"""

        return self.re_mora.findall(hira_text)

    def _kana_moraWakachi(self, kan_text):
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）

        self.cond = '('+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        return re_mora.findall(kana_text)

    def get7lists_from_orth(self, orth_wrd):
        """書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
        書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る"""

        # 単語の表層形を，読みに変換して `_yomi` に格納
        # ntt_orth2hira という命名はおかしかったから修正 2022_0309
        if orth_wrd in self.ntt_orth2hira:
            _yomi = self.ntt_orth2hira[orth_wrd]
        else:
            _yomi = jaconv.kata2hira(self.yomi(orth_wrd).strip())

        # `_yomi` を julius 表記に変換して `_phon` に代入
        _phon = self.hira2julius(_yomi)  # .split(' ')

        # 2023_0206 追加
        if orth_wrd == '九〇':
            _phon = ['ky', 'u:', 'j', 'u:']
            _yomi = 'きゅうじゅう'
        if orth_wrd == '八〇':
            _phon = ['h', 'a', 'ch', 'i', 'j', 'u:']
            _yomi = 'はちじゅう'
        if orth_wrd == '五〇':
            _phon = ['g', 'o', 'j', 'u:']
            _yomi = 'ごじゅう'
        if orth_wrd == '一〇':
            _phon = ['j', 'u:']
            _yomi = 'じゅう'
        if orth_wrd == '二〇':
            _phon = ['n', 'i', 'j', 'u:']
            _yomi = 'にじゅう'
        if orth_wrd == '六〇':
            _phon = ['r', 'o', 'k', 'u', 'j', 'u:']
            _yomi = 'ろくじゅう'
        if orth_wrd == '二〇〇〇':
            _phon = ['n', 'i', 's', 'e', 'N']
            _yomi = 'にせん'
        if orth_wrd == '三〇':
            _phon = ['s', 'a', 'N', 'j', 'u:']
            _yomi = 'さんじゅう'
        if orth_wrd == '七〇':
            _phon = ['n', 'a', 'n', 'a', 'j', 'u:']
            _yomi = 'ななじゅう'
        if orth_wrd == '四〇':
            _phon = ['y', 'o', 'N', 'j', 'u:']
            _yomi = 'よんじゅう'
        if orth_wrd == '一九九〇':
            _phon = ['s', 'e', 'N', 'ky', 'u:', 'hy',
                     'a', 'k', 'u', 'ky', 'u:', 'j', 'u:']
            _yomi = 'せんきゅうひゃくきゅうじゅう'
        if orth_wrd == '1丁目':
            _phon = ['i', 'q', 'ch', 'o:', 'm', 'e']
            _yomi = 'いっちょうめ'
        if orth_wrd == '2丁目':
            _phon = ['n', 'i', 'ch', 'o:', 'm', 'e']
            _yomi = 'にちょうめ'
        if orth_wrd == '一〇〇':
            _phon = ['hy', 'a', 'k', 'u']
            _yomi = 'ひゃく'
        if orth_wrd == '3丁目':
            _phon = ['s', 'a', 'N', 'ch', 'o:', 'm', 'e']
            _yomi = 'さんちょうめ'
        if orth_wrd == '2階':
            _phon = ['n', 'i', 'k', 'a', 'i']
            _yomi = 'にかい'
        if orth_wrd == '一九八〇':
            _phon = ['s', 'e', 'N', 'ky', 'u:', 'hy', 'a',
                     'k', 'u', 'h', 'a', 'ch', 'i', 'j', 'u:']
            _yomi = 'せんきゅうひゃくはちじゅう'
        if orth_wrd == '第1号':
            _phon = ['d', 'a', 'i:', 'ch', 'i', 'g', 'o:']
            _yomi = 'だいいちごう'
        if orth_wrd == '4丁目':
            _phon = ['y', 'o', 'N', 'ch', 'o:', 'm', 'e']
            _yomi = 'よんちょうめ'
        if orth_wrd == '1部':
            _phon = ['i', 'ch', 'i', 'b', 'u']
            _yomi = 'いちぶ'
        if orth_wrd == '一九七〇':
            _phon = ['s', 'e', 'N', 'ky', 'u:', 'hy', 'a',
                     'k', 'u', 'n', 'a', 'n', 'a', 'j', 'u:']
            _yomi = 'せんきゅうひゃくななじゅう'
        if orth_wrd == '3区':
            _phon = ['s', 'a', 'N', 'k', 'u']
            _yomi = 'さんく'
        if orth_wrd == '一九六〇':
            _phon = ['s', 'e', 'N', 'ky', 'u:', 'hy', 'a',
                     'k', 'u', 'r', 'o', 'k', 'u', 'j', 'u:']
            _yomi = 'せんきゅうひゃくろくじゅう'
        if orth_wrd == '第5':
            _phon = ['d', 'a', 'i', 'g', 'o']
            _yomi = 'だいご'
        if orth_wrd == '1号':
            _phon = ['i', 'ch', 'i', 'g', 'o:']
            _yomi = 'いちごう'
        if orth_wrd == '二〇一〇':
            _phon = ['n', 'i', 's', 'e', 'N', 'j', 'u:']
            _yomi = 'にせんじゅう'
        if orth_wrd == '2号':
            _phon = ['n', 'i', 'g', 'o:']
            _yomi = 'にごう'
        if orth_wrd == '2部':
            _phon = ['n', 'i', 'b', 'u']
            _yomi = 'にぶ'
        if orth_wrd == '3階':
            _phon = ['s', 'a', 'N', 'k', 'a', 'i']
            _yomi = 'さんかい'
        if orth_wrd == '第6':
            _phon = ['d', 'a', 'i', 'r', 'o', 'k', 'u']
            _yomi = 'だいろく'
        if orth_wrd == '一九五〇':
            _phon = ['s', 'e', 'N', 'ky', 'u:', 'hy',
                     'a', 'k', 'u', 'g', 'o', 'j', 'u:']
            _yomi = 'せんきゅうひゃくごじゅう'
        if orth_wrd == '4区':
            _phon = ['y', 'o', 'N', 'k', 'u']
            _yomi = 'よんく'
        if orth_wrd == '4階':
            _phon = ['y', 'o', 'N', 'k', 'a', 'i']
            _yomi = 'よんかい'
        if orth_wrd == '3号':
            _phon = ['s', 'a', 'N', 'g', 'o:']
            _yomi = 'さんごう'
        if orth_wrd == '6丁目':
            _phon = ['r', 'o', 'k', 'u', 'ch', 'o:', 'm', 'e']
            _yomi = 'ろくちょうめ'
        if orth_wrd == '二〇〇':
            _phon = ['n', 'i', 'hy', 'a', 'k', 'u']
            _yomi = 'にひゃく'
        if orth_wrd == '3部':
            _phon = ['s', 'a', 'N', 'b', 'u']
            _yomi = 'さんぶ'
        if orth_wrd == '5区':
            _phon = ['g', 'o', 'k', 'u']
            _yomi = 'ごく'
        if orth_wrd == '三〇〇':
            _phon = ['s', 'a', 'N', 'by', 'a', 'k', 'u']
            _yomi = 'さんびゃく'
        if orth_wrd == '11階':
            _phon = ['j', 'u:', 'i', 'q', 'k', 'a', 'i']
            _yomi = 'じゅういっかい'
        if orth_wrd == '4号':
            _phon = ['y', 'o', 'N', 'g', 'o:']
            _yomi = 'よんごう'
        if orth_wrd == '23区':
            _phon = ['n', 'i', 'j', 'u:', 's', 'a', 'N', 'k', 'u']
            _yomi = 'にじゅうさんく'
        if orth_wrd == '8階':
            _phon = ['h', 'a', 'q', 'k', 'a', 'i']
            _yomi = 'はっかい'
        if orth_wrd == '5階':
            _phon = ['g', 'o', 'k', 'a', 'i']
            _yomi = 'ごかい'
        if orth_wrd == '7階':
            _phon = ['n', 'a', 'n', 'a', 'k', 'a', 'i']
            _yomi = 'ななかい'
        if orth_wrd == '一一〇':
            _phon = ['hy', 'a', 'k', 'u', 'j', 'u:']
            _yomi = 'ひゃくじゅう'
        if orth_wrd == '1階':
            _phon = ['i', 'q', 'k', 'a', 'i']
            _yomi = 'いっかい'
        if orth_wrd == '1区':
            _phon = ['i', 'q', 'k', 'u']
            _yomi = 'いっく'
        if orth_wrd == '2区':
            _phon = ['n', 'i', 'k', 'u']
            _yomi = 'にく'
        if orth_wrd == '5丁目':
            _phon = ['g', 'o', 'ch', 'o:', 'm', 'e']
            _yomi = 'ごちょうめ'

        for _phn in _phon:
            if not _phn in self.phon_vocab:
                __yomi = self.yomi(_phn).strip()
                __hira = jaconv.kata2hira(__yomi).strip()
                __juli = jaconv.hiragana2julius(__hira).split()
                _phon = __juli

        # if orth_wrd == '二十': _phon = ['n','i','jy','u']; _yomi = 'にじゅう'
        # if orth_wrd == '二十一': _phon = ['二十一']
        # if orth_wrd == '十四': _phon =['十四']
        # if orth_wrd == '五十': _phon: = ['五十']
        # if orth_wrd == '二十五': _phon = ['二十五']
        # if orth_wrd == '十七': _phon = ['十七']
        # if orth_wrd == '二十二': _phon =  ['二十二']
        # if orth_wrd == '二十七': _phon = ['二十七']
        # if orth_wrd == '二十八': _phon = ['二十八']
        # if orth_wrd == '二十六': _phon = ['二十六']
        # if orth_wrd == '二十三': _phon = ['二十三']
        # if orth_wrd == '二十九': _phon = ['二十九']
        # if orth_wrd == '九二': _phon = ['九二']
        # if orth_wrd == '九五': _phon = ['九五']
        # if orth_wrd == '四十': _phon = ['四十']
        # if orth_wrd == '三十一': _phon = ['三十一']
        # if orth_wrd == '二百': _phon = ['二百']
        # if orth_wrd == '六十': _phon = ['六十']
        # if orth_wrd == '九一': _phon = ['九一']
        # if orth_wrd == '五百': _phon = ['五百']
        # if orth_wrd == '三百': _phon = ['三百']
        # if orth_wrd == '十万': _phon = ['十万']
        # if orth_wrd == '九六': _phon = ['九六']
        # if orth_wrd == '〇三': _phon = ['〇三']
        # if orth_wrd == '一九九四': _phon = ['一九九四']
        # if orth_wrd == '二千': _phon = ['二千']
        # if orth_wrd == '一九九五': _phon = ['一九九五']
        # if orth_wrd == '一九九三': _phon = ['一九九三']
        # if orth_wrd == '三千': _phon = ['三千']
        # if orth_wrd == '七十': _phon = ['七十']
        # if orth_wrd == '八九': _phon = ['八九']
        # if orth_wrd == '一九九二': _phon = ['一九九二']
        # if orth_wrd == '五千': _phon = ['五千']
        # if orth_wrd == '二五': _phon = ['二五']
        # if orth_wrd == '八十': _phon = ['八十']
        # if orth_wrd == '四百': _phon = ['四百']
        # if orth_wrd == '九七': _phon = ['九七']
        # if orth_wrd == '一千万': _phon = ['一千万']
        # if orth_wrd == '一五': _phon = ['一五']
        # if orth_wrd == '一九九六': _phon = ['一九九六']
        # if orth_wrd == '一億': _phon = ['一億']
        # if orth_wrd == '一九九一': _phon = ['一九九一']
        # if orth_wrd == '八五': _phon = ['八五']
        # if orth_wrd == '八八': _phon = ['八八']
        # if orth_wrd == '四五': _phon = ['四五']
        # if orth_wrd == '四十五': _phon = ['四十五']
        # if orth_wrd == '二万': _phon = ['二万']
        # if orth_wrd == '八七': _phon = ['八七']
        # if orth_wrd == '三五': _phon = ['三五']
        # if orth_wrd == '七五': _phon = ['七五']
        # if orth_wrd == '八六': _phon = ['八六']
        # if orth_wrd == '四八': _phon = ['四八']
        # if orth_wrd == '百五十': _phon = ['百五十']
        # if orth_wrd == '十億': _phon = ['十億']
        # if orth_wrd == '四九': _phon = ['四九']
        # if orth_wrd == '六五': _phon = ['六五']
        # if orth_wrd == '四六': _phon = ['四六']
        # if orth_wrd == '五五': _phon = ['五五']
        # if orth_wrd == '四七': _phon = ['四七']
        # if orth_wrd == '千五百': _phon = ['千五百']
        # if orth_wrd == '五一': _phon = ['五一']
        # if orth_wrd == '一九九七': _phon = ['一九九七']
        # if orth_wrd == '九八': _phon = ['九八']
        # if orth_wrd == '一九八九': _phon = ['一九八九']
        # if orth_wrd == '五万': _phon = ['五万']
        # if orth_wrd == '五三': _phon = ['五三']
        # if orth_wrd == '六百': _phon = ['六百']
        # if orth_wrd == '三十二': _phon = ['三十二']
        # if orth_wrd == '四四': _phon = ['四四']
        # if orth_wrd == '五四': _phon = ['五四']
        # if orth_wrd == '四千': _phon = ['四千']
        # if orth_wrd == '五二': _phon = ['五二']
        # if orth_wrd == '三万': _phon = ['三万']
        # if orth_wrd == '八四': _phon = ['八四']
        # if orth_wrd == '六十五': _phon = ['六十五']
        # if orth_wrd == '三二': _phon = ['三二']
        # if orth_wrd == '二七': _phon = ['二七']
        # if orth_wrd == '三八': _phon = ['三八']
        # if orth_wrd == '二九': _phon = ['二九']
        # if orth_wrd == '三一': _phon = ['三一']
        # if orth_wrd == '三三': _phon = ['三三']
        # if orth_wrd == '三七': _phon = ['三七']
        # if orth_wrd == '二八': _phon = ['二八']
        # if orth_wrd == '二二': _phon = ['二二']
        # if orth_wrd == '五七': _phon = ['五七']
        # if orth_wrd == '六一': _phon = ['六一']
        # if orth_wrd == '四二': _phon = ['四二']
        # if orth_wrd == '三六': _phon = ['三六']
        # if orth_wrd == '一三': _phon = ['一三']
        # if orth_wrd == '五九': _phon = ['五九']
        # if orth_wrd == '二百万': _phon = ['二百万']
        # if orth_wrd == '六八': _phon = ['六八']
        # if orth_wrd == '六四': _phon = ['六四']
        # if orth_wrd == '五十万': _phon = ['五十万']
        # if orth_wrd == '二四': _phon = ['二四']
        # if orth_wrd == '四一': _phon = ['四一']
        # if orth_wrd == '四三': _phon = ['四三']
        # if orth_wrd == '二三': _phon = ['二三']
        # if orth_wrd == '二一': _phon = ['二一']
        # if orth_wrd == '九十': _phon = ['九十']
        # if orth_wrd == '一二': _phon = ['一二']
        # if orth_wrd == '二十万': _phon = ['二十万']
        # if orth_wrd == '三四': _phon = ['三四']
        # if orth_wrd == '一九九八': _phon = ['一九九八']
        # if orth_wrd == '六七': _phon = ['六七']
        # if orth_wrd == '八百': _phon = ['八百']
        # if orth_wrd == '三十万': _phon = ['三十万']
        # if orth_wrd == '百億': _phon = ['百億']
        # if orth_wrd == '八二': _phon = ['八二']
        # if orth_wrd == '一四': _phon = ['一四']
        # if orth_wrd == '六三': _phon = ['六三']
        # if orth_wrd == '一七': _phon = ['一七']
        # if orth_wrd == '三百万': _phon = ['三百万']
        # if orth_wrd == '一九': _phon = ['一九']
        # if orth_wrd == '六六': _phon = ['六六']
        # if orth_wrd == '七二': _phon = ['七二']
        # if orth_wrd == '一一': _phon = ['一一']
        # if orth_wrd == '一六': _phon = ['一六']
        # if orth_wrd == '三十三': _phon = ['三十三']
        # if orth_wrd == '二千万': _phon = ['二千万']
        # if orth_wrd == '三十六': _phon = ['三十六']
        # if orth_wrd == '三十四': _phon = ['三十四']
        # if orth_wrd == '五百万': _phon = ['五百万']
        # if orth_wrd == '七四': _phon = ['七四']
        # if orth_wrd == '一九八八': _phon = ['一九八八']
        # if orth_wrd == '六九': _phon = ['六九']
        # if orth_wrd == '七一': _phon = ['七一']
        # if orth_wrd == '六千': _phon = ['六千']
        # if orth_wrd == '百二十': _phon = ['百二十']
        # if orth_wrd == '三十八': _phon = ['三十八']
        # if orth_wrd == '七六': _phon = ['七六']
        # if orth_wrd == '七九': _phon = ['七九']
        # if orth_wrd == '七八': _phon = ['七八']
        # if orth_wrd == '五十五': _phon = ['五十五']
        # if orth_wrd == '四十八': _phon = ['四十八']
        # if orth_wrd == '二百五十': _phon = ['二百五十']
        # if orth_wrd == '一九八七': _phon = ['一九八七']
        # if orth_wrd == '三十七': _phon = ['三十七']
        # if orth_wrd == '四十七': _phon = ['四十七']
        # if orth_wrd == '七七': _phon = ['七七']
        # if orth_wrd == '二千五百': _phon = ['二千五百']
        # if orth_wrd == '二億': _phon = ['二億']
        # if orth_wrd == '五十八': _phon = ['五十八']
        # if orth_wrd == '四十四': _phon = ['四十四']
        # if orth_wrd == '一九八六': _phon = ['一九八六']
        # if orth_wrd == '四十六': _phon = ['四十六']
        # if orth_wrd == '一九八五': _phon = ['一九八五']
        # if orth_wrd == '二〇〇二': _phon = ['二〇〇二']
        # if orth_wrd == '一九九九': _phon = ['一九九九']
        # if orth_wrd == '千二百': _phon = ['千二百']
        # if orth_wrd == '四十三': _phon = ['四十三']
        # if orth_wrd == '五億': _phon = ['五億']
        # if orth_wrd == '二〇〇一': _phon = ['二〇〇一']
        # if orth_wrd == '三千万': _phon = ['三千万']
        # if orth_wrd == '四十一': _phon = ['四十一']
        # if orth_wrd == '五千万': _phon = ['五千万']
        # if orth_wrd == '三十九': _phon = ['三十九']
        # if orth_wrd == '一万五千': _phon = ['一万五千']
        # if orth_wrd == '五十九': _phon = ['五十九']
        # if orth_wrd == '七千': _phon = ['七千']
        # if orth_wrd == '八千': _phon = ['八千']
        # if orth_wrd == '一兆': _phon = ['一兆']
        # if orth_wrd == ':四十九': _phon = ['四十九']
        # if orth_wrd == ':五十二': _phon = ['五十二']
        # if orth_wrd == ':百三十': _phon = ['百三十']
        # if orth_wrd == ':七十五': _phon = ['七十五']
        # if orth_wrd == ':五十七': _phon = ['五十七']
        # if orth_wrd == ':五十一': _phon = ['五十一']
        # if orth_wrd == ':六十四': _phon = ['六十四']
        # if orth_wrd == ':三億': _phon = ['三億']
        # if orth_wrd == ':六万': _phon = ['六万']
        # if orth_wrd == ':五十三': _phon = ['五十三']
        # if orth_wrd == ':五十四': _phon = ['五十四']
        # if orth_wrd == ':一九八四': _phon = ['一九八四']
        # if orth_wrd == ':二十億': _phon = ['二十億']
        # if orth_wrd == ':一千': _phon = ['一千']
        # if orth_wrd == ':三千五百': _phon = ['三千五百']
        # if orth_wrd == ':十五万': _phon = ['十五万']
        # if orth_wrd == ':一九八三': _phon = ['一九八三']
        # if orth_wrd == ':一千億': _phon = ['一千億']
        # if orth_wrd == ':二兆': _phon = ['二兆']
        # if orth_wrd == ':一万二千': _phon = ['一万二千']
        # if orth_wrd == ':三十億': _phon = ['三十億']
        # if orth_wrd == ':六十一': _phon = ['六十一']
        # if orth_wrd == ':二百億': _phon = ['二百億']
        # if orth_wrd == ':百五十万': _phon = ['百五十万']
        # if orth_wrd == ':四百万': _phon = ['四百万']
        # if orth_wrd == ':三百億': _phon = ['三百億']
        # if orth_wrd == ':四千万': _phon = ['四千万']
        # if orth_wrd == ':千八百': _phon = ['千八百']
        # if orth_wrd == ':千三百': _phon = ['千三百']
        # if orth_wrd == ':一九八二': _phon = ['一九八二']
        # if orth_wrd == ':百十': _phon = ['百十']
        # if orth_wrd == ':一九七九': _phon = ['一九七九']
        # if orth_wrd == ':三百五十': _phon = ['三百五十']
        # if orth_wrd == ':六十万': _phon = ['六十万']
        # if orth_wrd == ':百四十': _phon = ['百四十']
        # if orth_wrd == ':七万': _phon = ['七万']
        # if orth_wrd == ':六百万': _phon = ['六百万']
        # if orth_wrd == ':百六十': _phon = ['百六十']
        # if orth_wrd == ':百八十': _phon = ['百八十']
        # if orth_wrd == ':七十二': _phon = ['七十二']
        # if orth_wrd == ':六十二': _phon = ['六十二']
        # if orth_wrd == ':五十億': _phon = ['五十億']
        # if orth_wrd == ':九百': _phon = ['九百']
        # if orth_wrd == ':一九四五': _phon = ['一九四五']
        # if orth_wrd == ':二千億': _phon = ['二千億']
        # if orth_wrd == ':一九八一': _phon = ['一九八一']
        # if orth_wrd == ':千億': _phon = ['千億']
        # if orth_wrd == ':十兆': _phon = ['十兆']
        # if orth_wrd == ':一九七二': _phon = ['一九七二']
        # if orth_wrd == ':二万五千': _phon = ['二万五千']
        # if orth_wrd == ':九千': _phon = ['九千']
        # if orth_wrd == ':一九七八': _phon = ['一九七八']
        # if orth_wrd == ':六十六': _phon = ['六十六']
        # if orth_wrd == ':六千万': _phon = ['六千万']
        # if orth_wrd == ':一九七五': _phon = ['一九七五']
        # if orth_wrd == ':五百億': _phon = ['五百億']
        # if orth_wrd == ':四億': _phon = ['四億']
        # if orth_wrd == ':六十三': _phon = ['六十三']
        # if orth_wrd == ':三〇一': _phon = ['三〇一']
        # if orth_wrd == ':六十八': _phon = ['六十八']
        # if orth_wrd == ':四十億': _phon = ['四十億']
        # if orth_wrd == ':六十七': _phon = ['六十七']
        # if orth_wrd == ':一万三千': _phon = ['一万三千']
        # if orth_wrd == ':二十五万': _phon = ['二十五万']
        # if orth_wrd == ':六十九': _phon = ['六十九']
        # if orth_wrd == ':千六百': _phon = ['千六百']
        # if orth_wrd == ':七十四': _phon = ['七十四']
        # if orth_wrd == ':七百万': _phon = ['七百万']
        # if orth_wrd == ':一九七六': _phon = ['一九七六']
        # if orth_wrd == ':百七十': _phon = ['百七十']
        # if orth_wrd == ':四千五百': _phon = ['四千五百']
        # if orth_wrd == ':七十万': _phon = ['七十万']
        # if orth_wrd == ':七十六': _phon = ['七十六']
        # if orth_wrd == ':八十五': _phon = ['八十五']
        # if orth_wrd == ':千五百万': _phon = ['千五百万']
        # if orth_wrd == ':一九七四': _phon = ['一九七四']
        # if orth_wrd == ':七十三': _phon = ['七十三']
        # if orth_wrd == ':六千八百五十億': _phon = ['六千八百五十億']
        # if orth_wrd == ':千四百': _phon = ['千四百']
        # if orth_wrd == ':八十八': _phon = ['八十八']
        # if orth_wrd == ':四百五十': _phon = ['四百五十']
        # if orth_wrd == ':三千億': _phon = ['三千億']
        # if orth_wrd == ':一九七三': _phon = ['一九七三']
        # if orth_wrd == ':二〇〇三': _phon = ['二〇〇三']
        # if orth_wrd == ':七十七': _phon = ['七十七']
        # if orth_wrd == ':八千万': _phon = ['八千万']
        # if orth_wrd == ':一九七一': _phon = ['一九七一']
        # if orth_wrd == ':十二万': _phon = ['十二万']
        # if orth_wrd == ':一九六八': _phon = ['一九六八']
        # if orth_wrd == ':一九六四': _phon = ['一九六四']
        # if orth_wrd == ':七十一': _phon = ['七十一']
        # if orth_wrd == ':一九七七': _phon = ['一九七七']
        # if orth_wrd == ':一九六五': _phon = ['一九六五']
        # if orth_wrd == ':二〇〇五': _phon = ['二〇〇五']
        # if orth_wrd == ':一九六七': _phon = ['一九六七']
        # if orth_wrd == ':四百億': _phon = ['四百億']
        # if orth_wrd == ':七十八': _phon = ['七十八']
        # if orth_wrd == ':六億': _phon = ['六億']
        # if orth_wrd == ':一億五千万': _phon = ['一億五千万']
        # if orth_wrd == ':八十二': _phon = ['八十二']
        # if orth_wrd == ':十五億': _phon = ['十五億']
        # if orth_wrd == ':一万六千': _phon = ['一万六千']
        # if orth_wrd == ':五兆': _phon = ['五兆']
        # if orth_wrd == ':二百四十': _phon = ['二百四十']
        # if orth_wrd == ':七億': _phon = ['七億']
        # if orth_wrd == ':千七百': _phon = ['千七百']
        # if orth_wrd == ':一九六九': _phon = ['一九六九']
        # if orth_wrd == ':六百億': _phon = ['六百億']
        # if orth_wrd == ':百五十億': _phon = ['百五十億']
        # if orth_wrd == ':八十四': _phon = ['八十四']
        # if orth_wrd == ':六兆': _phon = ['六兆']
        # if orth_wrd == ':千百': _phon = ['千百']
        # if orth_wrd == ':八億': _phon = ['八億']
        # if orth_wrd == ':七十九': _phon = ['七十九']
        # if orth_wrd == ':二百五十万': _phon = ['二百五十万']
        # if orth_wrd == ':八十三': _phon = ['八十三']
        # if orth_wrd == ':七千万': _phon = ['七千万']
        # if orth_wrd == ':百二十万': _phon = ['百二十万']
        # if orth_wrd == ':八十一': _phon = ['八十一']
        # if orth_wrd == ':十二億': _phon = ['十二億']
        # if orth_wrd == ':五千億': _phon = ['五千億']
        # if orth_wrd == ':六十億': _phon = ['六十億']
        # if orth_wrd == ':千二百万': _phon = ['千二百万']
        # if orth_wrd == ':一九五五': _phon = ['一九五五']
        # if orth_wrd == ':四兆': _phon = ['四兆']
        # if orth_wrd == ':一億二千万': _phon = ['一億二千万']
        # if orth_wrd == ':〇六': _phon = ['〇六']
        # if orth_wrd == ':一万八千': _phon = ['一万八千']
        # if orth_wrd == ':九十五': _phon = ['九十五']
        # if orth_wrd == ':一九六六': _phon = ['一九六六']
        # if orth_wrd == ':三兆': _phon = ['三兆']
        # if orth_wrd == ':一万四千': _phon = ['一万四千']

        # 直上の `_phon` の逆転を作成して `_phone_r` に代入
        _phon_r = [_p_id for _p_id in _phon[::-1]]

        # 書記素をリストに変換
        _orth = [c for c in orth_wrd]

        # 直上の `_orth` の逆転を作成して `_orth_r` に代入
        _orth_r = [c for c in _orth[::-1]]

        #_mora = self.moraWakachi(jaconv.hira2kata(_yomi))
        _mora = self.moraWakachi(_yomi)

        for _m in _mora:
            if not _m in self.mora_vocab:
                self.mora_vocab.append(_m)
            for _j in self.hira2julius(_m):
                if not _j in self.mora_p:
                    self.mora_p[_j] = 1
                else:
                    self.mora_p[_j] += 1
        _mora_r = [_m for _m in _mora[::-1]]
        return _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r

    def hira2julius(self, text: str) -> str:
        """`jaconv.hiragana2julius()` では未対応の表記を扱う"""
        text = text.replace('ゔぁ', ' b a')
        text = text.replace('ゔぃ', ' b i')
        text = text.replace('ゔぇ', ' b e')
        text = text.replace('ゔぉ', ' b o')
        text = text.replace('ゔゅ', ' by u')

        #text = text.replace('ぅ゛', ' b u')
        text = jaconv.hiragana2julius(text).split()
        return text

    def set_source_and_target_from_params(
            self,
            source: str = 'orth',
            target: str = 'phon',
            is_print: bool = True):

        # ソースとターゲットを設定しておく

        if source == 'orth':
            self.source_vocab = self.orth_vocab
            self.source_ids = 'orth_ids'
            self.source_maxlen = self.max_orth_length
            self.source_ids2tkn = self.orth_ids2tkn
            self.source_tkn2ids = self.orth_tkn2ids
        elif source == 'phon':
            self.source_vocab = self.phon_vocab
            self.source_ids = 'phon_ids'
            self.source_maxlen = self.max_phon_length
            self.source_ids2tkn = self.phon_ids2tkn
            self.source_tkn2ids = self.phon_tkn2ids
        elif source == 'mora':
            self.source_vocab = self.mora_vocab
            self.source_ids = 'mora_ids'
            self.source_maxlen = self.max_mora_length
            #self.source_ids2tkn = self.mora_ids2tkn
            #self.source_tkn2ids = self.mora_tkn2ids
        elif source == 'mora_p':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_ids2tkn
            #self.source_tkn2ids = self.mora_p_tkn2ids
        elif source == 'mora_p_r':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids_r'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_r_ids2tkn
            #self.source_tkn2ids = self.mora_p_r_tkn2ids

        if target == 'orth':
            self.target_vocab = self.orth_vocab
            self.target_ids = 'orth_ids'
            self.target_maxlen = self.max_orth_length
            self.target_ids2tkn = self.orth_ids2tkn
            self.target_tkn2ids = self.orth_tkn2ids
        elif target == 'phon':
            self.target_vocab = self.phon_vocab
            self.target_ids = 'phon_ids'
            self.target_maxlen = self.max_phon_length
            self.target_ids2tkn = self.phon_ids2tkn
            self.target_tkn2ids = self.phon_tkn2ids
        elif target == 'mora':
            self.target_vocab = self.mora_vocab
            self.target_ids = 'mora_ids'
            self.target_maxlen = self.max_mora_length
            #self.target_ids2tkn = self.mora_ids2tkn
            #self.target_tkn2ids = self.mora_tkn2ids
        elif target == 'mora_p':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_ids2tkn
            #self.target_tkn2ids = self.mora_p_tkn2ids
        elif target == 'mora_p_r':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids_r'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_r_ids2tkn
            #self.target_tkn2ids = self.mora_p__r_tkn2ids

        if is_print:
            print(colored(f'self.source:{self.source}', 'blue',
                          attrs=['bold']), f'{self.source_vocab}')
            print(colored(f'self.target:{target}', 'cyan',
                          attrs=['bold']), f'{self.target_vocab}')
            print(colored(f'self.source_ids:{self.source_ids}',
                          'blue', attrs=['bold']), f'{self.source_ids}')
            print(colored(f'self.target_ids:{self.target_ids}',
                          'cyan', attrs=['bold']), f'{self.target_ids}')

        return  # source_vocab, source_ids, target_vocab, target_ids

    def __len__(self) -> int:
        return len(self.train_data)

    def __call__(self, x: int) -> dict:
        return self.train_data[x][self.source_ids], self.train_data[x][self.target_ids]

    def __getitem__(self, x: int) -> dict:

        # return self.train_data[x][self.source_ids], self.train_data[x][self.target_ids]
        return self.order[x][self.source_ids] + [self.source_vocab.index('<EOW>')], self.order[x][self.target_ids] + [self.target_vocab.index('<EOW>')]

    def set_train_vocab(self):
        # def set_train_vocab_minus_test_vocab(self):
        """JISX2008-1990 コードから記号とみなしうるコードを集めて ja_symbols とする
        記号だけから構成されている word2vec の項目は排除するため
        """
        self.ja_symbols = '、。，．・：；？！゛゜´\' #+ \'｀¨＾‾＿ヽヾゝゞ〃仝々〆〇ー—‐／＼〜‖｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋−±×÷＝≠＜＞≦≧∞∴♂♀°′″℃¥＄¢£％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨¬⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪†‡¶◯#ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        self.ja_symbols_normalized = jaconv.normalize(self.ja_symbols)

        print(f'# 訓練に用いる単語の選定 {self.traindata_size} 語')
        vocab = []
        i = 0
        while i < len(self.ntt_freq):
            word = self.ntt_freq[i]
            if word == '\u3000':  # NTT 日本語の語彙特性で，これだけ変なので特別扱い
                i += 1
                continue

            # 良い回避策が見つからないので，以下の行の変換だけ特別扱いしている
            #word = jaconv.normalize(word).replace('・', '').replace('ヴ', 'ブ')
            # word の normalized しないでみる
            word = word.replace('・', '').replace('ヴ', 'ブ')

            # if not word.isascii():
            #    if not word in vocab:
            #        vocab.append(word)
            #        if len(vocab) >= self.traindata_size:
            #            return vocab
            # and (word in self.w2v):
            if (not word in self.ja_symbols) and (not word.isascii()):
                if not word in vocab:
                    vocab.append(word)
                    if len(vocab) >= self.traindata_size:
                        return vocab
            i += 1
        return vocab

    def make_ntt_freq_data(self,
                           ps71_fname: str = None):

        print('# NTT日本語語彙特性 (天野，近藤; 1999, 三省堂)より頻度情報を取得')

        if ps71_fname == None:
            # データファイルの保存してあるディレクトリの指定
            ntt_dir = 'ccap'
            psy71_fname = 'psylex71utf8.txt'  # ファイル名
            psy71_fname = 'psylex71utf8.txt.gz'  # ファイル名
            # with gzip.open(os.path.join(ntt_dir,psy71_fname), 'r') as f:
            with gzip.open(os.path.join(ntt_dir, psy71_fname), 'rt', encoding='utf-8') as f:
                ntt71raw = f.readlines()
        else:
            with open(ps71_fname, 'r') as f:
                ntt71raw = f.readlines()

        tmp = [line.split(' ')[:6] for line in ntt71raw]
        tmp2 = [[int(line[0]), line[2], line[4], int(line[5]), line[3]]
                for line in tmp]
        # 単語ID(0), 単語，品詞，頻度 だけ取り出す

        ntt_freq = {x[0]-1: {'単語': jaconv.normalize(x[1]),
                             '品詞': x[2],
                             '頻度': x[3],
                             'よみ': jaconv.kata2hira(jaconv.normalize(x[4]))
                             } for x in tmp2}
        #ntt_freq = {x[0]-1:{'単語':x[1],'品詞':x[2],'頻度':x[3], 'よみ':x[4]} for x in tmp2}
        ntt_orth2hira = {ntt_freq[x]['単語']: ntt_freq[x]['よみ'] for x in ntt_freq}
        # print(f'#登録総単語数: {len(ntt_freq)}')

        Freq = np.zeros((len(ntt_freq)), dtype=np.uint)  # ソートに使用する numpy 配列
        for i, x in enumerate(ntt_freq):
            Freq[i] = ntt_freq[i]['頻度']

        Freq_sorted = np.argsort(Freq)[::-1]  # 頻度降順に並べ替え

        # self.ntt_freq には頻度順に単語が並んでいる
        return [ntt_freq[x]['単語']for x in Freq_sorted], ntt_orth2hira
