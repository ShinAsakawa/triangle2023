import torch
import jaconv
from termcolor import colored

device="cuda:0" if torch.cuda.is_available() else "cpu"

class Onechar_dataset(torch.utils.data.Dataset):
    """一文字データセットの定義"""

    def __init__(self,
                 source:str='orth',  # 'phon'
                 target:str='phon',  # 'orth'
				 yomi=None,
                 _vocab=None):

        super().__init__()

        _src = source
        _tgt = target

        _src = 'mora' if _src == 'mora_p' else _src
        _tgt = 'mora' if _tgt == 'mora_p' else _tgt
        _src, _tgt = _src+'_ids', _tgt+'_ids'

        self.source = source
        self.target = target

        num = '０１２３４５６７８９'
        alpha = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'   # ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        hira = 'あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをん'
        kata = 'アイウエオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモヤユヨラリルレロワヰヱヲン'
        onechars = hira+alpha+num  # +kata

        # digit_alpha = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'  # ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        # hira = 'あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをん'
        # kata = 'アイウエオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモヤユヨラリルレロワヰヱヲン'
        # onechars = digit_alpha+hira  # +kata

        self.phon_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
                           'N', 'a', 'a:', 'e', 'e:', 'i', 'i:', 'i::', 'o', 'o:', 'o::', 'u',
                           'u:', 'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy', 'j', 'k', 'ky',
                           'm', 'my', 'n', 'ny', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z']

        self.orth_vocab =  ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        for ch in onechars:
            self.orth_vocab.append(ch)

        max_orth_length, max_phon_length, max_mora_length, max_mora_p_length = 0, 0, 0, 0
        data_dict = {}
        for _orth in onechars:
            i = len(data_dict)
            _yomi = yomi(_orth).strip()
            _hira = jaconv.kata2hira(_yomi)
            _phon = jaconv.hiragana2julius(_hira).split(' ')
            #phon_ids = [self.phon_vocab.index(p) if p in self.phon_vocab else self.phon_vocab.index('<UNK>') for p in _phon_juli]
            phon_ids = _vocab.phon_tkn2ids(_phon)
            orth_ids = _vocab.orth_tkn2ids(_orth)

            _data = {'_yomi': _yomi,
                     'phon': _phon,
                     'phon_ids': phon_ids,
                     'orth': _orth,
                     'orth_ids': orth_ids}
            __src, __tgt = _data[_src], _data[_tgt]
            data_dict[i] = {'yomi': _yomi,
                            'orth': _orth,
                            'src': __src,
                            'tgt': __tgt,
                            '_phon': _phon,
                            'phon_ids': phon_ids,
                            'orth_ids': orth_ids}
            len_orth, len_phon = len(_orth), len(_phon)
            max_orth_length = len_orth if len_orth > max_orth_length else max_orth_length
            max_phon_length = len_phon if len_phon > max_phon_length else max_phon_length
            #max_mora_length = len_mora if len_mora > max_mora_length else max_mora_length
            #max_mora_p_length = len_mora_p if len_mora_p > max_mora_p_length else max_mora_p_length

        self.max_orth_length = max_orth_length
        self.max_phon_length = max_phon_length
        self.data_dict = data_dict
        self.set_source_and_target_from_params()

    def __len__(self) -> int:
        return len(self.data_dict)

    def __getitem__(self, x: int):
        _data = self.data_dict[x]
        return _data['src'], _data['tgt']

    def __call__(self, x: int) -> dict:
        return self.data_dict[x][self.source_ids], self.data_dict[x][self.target_ids]

    def __getitem__(self, x: int) -> dict:
        #return self.train_data[x][self.source_ids], self.train_data[x][self.target_ids]
        _src = self.data_dict[x][self.source_ids] + [self.source_vocab.index('<EOW>')]
        #_src = list(self.data_dict[x][self.source_ids]) + [self.source_vocab.index('<EOW>')]
        _tgt = self.data_dict[x][self.target_ids] + [self.target_vocab.index('<EOW>')]
        #_tgt = [self.data_dict[x][self.target_ids]]
        #print(f'type(_src):{type(_src)}',
        #      f'type(_tgt):{type(_tgt)}')
        return _src, _tgt
        #return list(self.data_dict[x][self.source_ids]) + [self.source_vocab.index('<EOW>')], self.data_dict[x][self.target_ids] + [self.target_vocab.index('<EOW>')]


    def set_source_and_target_from_params(self, is_print:bool = True):

        # ソースとターゲットを設定しておく
        if self.source == 'orth':
            self.source_vocab = self.orth_vocab
            self.source_ids = 'orth_ids'
            self.source_maxlen = self.max_orth_length
            self.source_ids2tkn = self.orth_ids2tkn
            self.source_tkn2ids = self.orth_tkn2ids
        elif self.source == 'phon':
            self.source_vocab = self.phon_vocab
            self.source_ids = 'phon_ids'
            self.source_maxlen = self.max_phon_length
            self.source_ids2tkn = self.phon_ids2tkn
            self.source_tkn2ids = self.phon_tkn2ids
        elif self.source == 'mora':
            self.source_vocab = self.mora_vocab
            self.source_ids = 'mora_ids'
            self.source_maxlen = self.max_mora_length
            #self.source_ids2tkn = self.mora_ids2tkn
            #self.source_tkn2ids = self.mora_tkn2ids
        elif self.source == 'mora_p':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_ids2tkn
            #self.source_tkn2ids = self.mora_p_tkn2ids
        elif self.source == 'mora_p_r':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids_r'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_r_ids2tkn
            #self.source_tkn2ids = self.mora_p_r_tkn2ids

        if self.target == 'orth':
            self.target_vocab = self.orth_vocab
            self.target_ids = 'orth_ids'
            self.target_maxlen = self.max_orth_length
            self.target_ids2tkn = self.orth_ids2tkn
            self.target_tkn2ids = self.orth_tkn2ids
        elif self.target == 'phon':
            self.target_vocab = self.phon_vocab
            self.target_ids = 'phon_ids'
            self.target_maxlen = self.max_phon_length
            self.target_ids2tkn = self.phon_ids2tkn
            self.target_tkn2ids = self.phon_tkn2ids
        elif self.target == 'mora':
            self.target_vocab = self.mora_vocab
            self.target_ids = 'mora_ids'
            self.target_maxlen = self.max_mora_length
            #self.target_ids2tkn = self.mora_ids2tkn
            #self.target_tkn2ids = self.mora_tkn2ids
        elif self.target == 'mora_p':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_ids2tkn
            #self.target_tkn2ids = self.mora_p_tkn2ids
        elif self.target == 'mora_p_r':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids_r'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_r_ids2tkn
            #self.target_tkn2ids = self.mora_p__r_tkn2ids

        if is_print:
            print(colored(f'self.source:{self.source}', 'blue',
                        attrs=['bold']), f'{self.source_vocab}')
            print(colored(f'self.target:{self.target}', 'cyan',
                        attrs=['bold']), f'{self.target_vocab}')
            print(colored(f'self.source_ids:{self.source_ids}',
                        'blue', attrs=['bold']), f'{self.source_ids}')
            print(colored(f'self.target_ids:{self.target_ids}',
                        'cyan', attrs=['bold']), f'{self.target_ids}')


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
