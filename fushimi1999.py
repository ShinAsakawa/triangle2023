import torch
from termcolor import colored


fushimi1999 = {
    'HF___consist__': ['戦争', '倉庫', '医学', '注意', '記念', '番号', '料理', '完全', '開始', '印刷',
                    '連続', '予約', '多少', '教員', '当局', '材料', '夕刊', '労働', '運送', '電池'],  # consistent, 'high-frequency words
    'HF___inconsist': ['反対', '失敗', '作品', '指定', '実験', '決定', '独占', '独身', '固定', '食品',
                        '表明', '安定', '各種', '役所', '海岸', '決算', '地帯', '道路', '安打', '楽団'],  # inconsistent, 'high-frequency words
    'HF___atypical_': ['仲間', '夫婦', '人間', '神経', '相手', '反発', '化粧', '建物', '彼女', '毛糸',
                        '場合', '台風', '夜間', '人形', '東西', '地元', '松原', '競馬', '大幅', '貸家'],  # inconsistent atypical, 'high-frequency words
    'LF___consist__': ['集計', '観察', '予告', '動脈', '理学', '信任', '任務', '返信', '医局', '低温',
                        '区別', '永続', '持続', '試練', '満開', '軍備', '製材', '銀貨', '急送', '改選'],  # consistent, 'low-frequency words
    'LF___inconsist': ['表紙', '指針', '熱帯', '作詞', '決着', '食費', '古代', '地形', '役場', '品種',
                        '祝福', '金銭', '根底', '接種', '経由', '郷土', '街路', '宿直', '曲折', '越境'],  # inconsistent, 'low-frequency words
    'LF___atypical_': ['強引', '寿命', '豆腐', '出前', '歌声', '近道', '間口', '風物', '面影', '眼鏡',
                        '居所', '献立', '小雨', '毛皮', '鳥居', '仲買', '頭取', '極上', '奉行', '夢路'],  # inconsistent atypical, 'low-frequency words
    'HFNW_consist__': ['集学', '信別', '製信', '運学', '番送', '電続', '完意', '軍開', '動選', '当働',
                        '予続', '倉理', '予少', '教池', '理任', '銀務', '連料', '開員', '注全', '記争'],  # consistent, 'high-character-frequency non-words
    'HFNW_inconsist': ['作明', '風行', '失定', '指団', '決所', '各算', '海身', '東発', '楽験', '作代',
                        '反原', '独対', '歌上', '反定', '独定', '場家', '安種', '経着', '決土', '松合'],  # inconsistent biased, 'high-character-frequency non-words
    'HFNW_ambiguous': ['表品', '実定', '人風', '神間', '相経', '人元', '小引', '指場', '毛所', '台手',
                        '間物', '道品', '出取', '建馬', '大婦', '地打', '化間', '面口', '金由', '彼間'],  # inconsistent ambiguous, 'high-character-frequency non-words
    'LFNW_consist__': ['急材', '戦刊', '返計', '印念', '低局', '労号', '満送', '永告', '試脈', '観備',
                        '材約', '夕局', '医庫', '任続', '医貨', '改練', '区温', '多始', '材刷', '持察'],  # consistent, 'low-character-frequency non-words
    'LFNW_inconsist': ['食占', '表底', '宿帯', '決帯', '古費', '安敗', '役針', '近命', '眼道', '豆立',
                        '街直', '固路', '郷種', '品路', '曲銭', '献居', '奉買', '根境', '役岸', '祝折'],  # inconsistent biased, 'low-character-frequency non-words
    'LFNW_ambiguous': ['食形', '接紙', '競物', '地詞', '強腐', '頭路', '毛西', '夜糸', '仲影', '熱福',
                        '寿前', '鳥雨', '地粧', '越種', '仲女', '極鏡', '夢皮', '居声', '貸形', '夫幅'],  # inconsistent ambiguous, 'low-character-frequency non-words
}

def _fushimi1999_list(verbose:bool = False):

    fushimi1999_list = []
    for k, v in fushimi1999.items():
        for _v in v:
            fushimi1999_list.append(_v)

    if verbose:
        print(colored('# Fushimi1999 データから，訓練データに含まれているデータを表示する',
                      'blue', attrs=['bold']))
        for i, wrd in enumerate(fushimi1999_list):
            if wrd in train_wordlist:
                color = 'blue'
                idx = train_wordlist.index(wrd)
            else:
                color = 'red'
                idx = -1
            print(colored((f'{i:3d} wrd:{wrd},idx:{idx:5d}',
                           f'orth_tkn2ids:{orth_tkn2ids(wrd)}',  # o[_tgt]
                           ), color=color, attrs=['bold']))
    print(f'fushimi1999_list:{fushimi1999_list}') if verbose else None

    return fushimi1999_list


# def _fushimi1999_list(verbose: bool = False):

#     fushimi1999 = {
#         'HF___consist__': ['戦争', '倉庫', '医学', '注意', '記念', '番号', '料理', '完全', '開始', '印刷',
#                            '連続', '予約', '多少', '教員', '当局', '材料', '夕刊', '労働', '運送', '電池'],  # consistent, 'high-frequency words
#         'HF___inconsist': ['反対', '失敗', '作品', '指定', '実験', '決定', '独占', '独身', '固定', '食品',
#                            '表明', '安定', '各種', '役所', '海岸', '決算', '地帯', '道路', '安打', '楽団'],  # inconsistent, 'high-frequency words
#         'HF___atypical_': ['仲間', '夫婦', '人間', '神経', '相手', '反発', '化粧', '建物', '彼女', '毛糸',
#                            '場合', '台風', '夜間', '人形', '東西', '地元', '松原', '競馬', '大幅', '貸家'],  # inconsistent atypical, 'high-frequency words
#         'LF___consist__': ['集計', '観察', '予告', '動脈', '理学', '信任', '任務', '返信', '医局', '低温',
#                            '区別', '永続', '持続', '試練', '満開', '軍備', '製材', '銀貨', '急送', '改選'],  # consistent, 'low-frequecy words
#         'LF___inconsist': ['表紙', '指針', '熱帯', '作詞', '決着', '食費', '古代', '地形', '役場', '品種',
#                            '祝福', '金銭', '根底', '接種', '経由', '郷土', '街路', '宿直', '曲折', '越境'],  # inconsistent, 'low-frequency words
#         'LF___atypical_': ['強引', '寿命', '豆腐', '出前', '歌声', '近道', '間口', '風物', '面影', '眼鏡',
#                            '居所', '献立', '小雨', '毛皮', '鳥居', '仲買', '頭取', '極上', '奉行', '夢路'],  # inconsistent atypical, 'low-frequncy words
#         'HFNW_consist__': ['集学', '信別', '製信', '運学', '番送', '電続', '完意', '軍開', '動選', '当働',
#                            '予続', '倉理', '予少', '教池', '理任', '銀務', '連料', '開員', '注全', '記争'],  # consistent, 'high-character-frequency nonwords
#         'HFNW_inconsist': ['作明', '風行', '失定', '指団', '決所', '各算', '海身', '東発', '楽験', '作代',
#                            '反原', '独対', '歌上', '反定', '独定', '場家', '安種', '経着', '決土', '松合'],  # inconsistent biased, 'high-character-frequency nonwords
#         'HFNW_ambiguous': ['表品', '実定', '人風', '神間', '相経', '人元', '小引', '指場', '毛所', '台手',
#                            '間物', '道品', '出取', '建馬', '大婦', '地打', '化間', '面口', '金由', '彼間'],  # inconsistent ambigous, 'high-character-frequency nonwords
#         'LFNW_consist__': ['急材', '戦刊', '返計', '印念', '低局', '労号', '満送', '永告', '試脈', '観備',
#                            '材約', '夕局', '医庫', '任続', '医貨', '改練', '区温', '多始', '材刷', '持察'],  # consistent, 'low-character-frequency nonwords
#         'LFNW_inconsist': ['食占', '表底', '宿帯', '決帯', '古費', '安敗', '役針', '近命', '眼道', '豆立',
#                            '街直', '固路', '郷種', '品路', '曲銭', '献居', '奉買', '根境', '役岸', '祝折'],  # inconsistent biased, 'low-character-frequency nonwords
#         'LFNW_ambiguous': ['食形', '接紙', '競物', '地詞', '強腐', '頭路', '毛西', '夜糸', '仲影', '熱福',
#                            '寿前', '鳥雨', '地粧', '越種', '仲女', '極鏡', '夢皮', '居声', '貸形', '夫幅'],  # inconsistent ambigous, 'low-character-frequency nonwords
#     }

#     fushimi1999_list = []
#     for k, v in fushimi1999.items():
#         for _v in v:
#             fushimi1999_list.append(_v)

#     if verbose:
#         print(colored('# Fushimi1999 データから，訓練データに含まれているデータを表示する',
#                       'blue', attrs=['bold']))
#         for i, wrd in enumerate(fushimi1999_list):
#             if wrd in train_wordlist:
#                 color = 'blue'
#                 idx = train_wordlist.index(wrd)
#             else:
#                 color = 'red'
#                 idx = -1
#             print(colored((f'{i:3d} wrd:{wrd},idx:{idx:5d}',
#                            f'orth_tkn2ids:{orth_tkn2ids(wrd)}',  # o[_tgt]
#                            ), color=color, attrs=['bold']))
#     print(f'fushimi1999_list:{fushimi1999_list}') if verbose else None

#     return fushimi1999_list


def check_fushimi_list(
    #wordlist:list=None,
    encoder:torch.nn.Module=None, #encoder,
    decoder:torch.nn.Module=None, # decoder,
    is_print:bool=False,
    )->str:

    ret_msg = ""
    counter = 1
    for key, val in fushimi1999.items():
        key_old = ""
        if key != key_old:
            key_old = key
            ret_msg += f'{key}:'
            if is_print:
                print(colored(f'{key}:', 'green', attrs=['bold']), end=" ")
        
        n_ok, n_all = 0, 0
        msg = ""
        for wrd in val:
            _orth = _vocab.orth_tkn2ids(wrd) + [_vocab.orth_vocab.index('<EOW>')]
            ans=evaluate(encoder,
                         decoder,
                         _orth,
                         _vocab.max_length,
                         source_vocab,
                         target_vocab,
                         source_ids,
                         target_ids)
            
            res = "".join(p for p in ans[0][:-1])  # モデルからの戻り値を再構成
            if res == wrd:
                n_ok += 1
            n_all += 1

            counter =  1 if (counter % 10) == 0 else (counter + 1)
            _end = "\n" if counter==1 else ", "
            msg += f"{wrd}->/{res}/{_end}"
            
        ret_msg += f'{n_ok/n_all * 100:5.2f}%\n{msg}'
        if is_print:
            print(f'{n_ok/n_all * 100:5.2f}%\n{msg}')
            
    return ret_msg
