import re
import os
import random
from glob import glob
from typing import List
from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm import tqdm


class SentTokenizer:
    SENT_RE = re.compile(r'[^\.?!…]+[\.?!…]*["»“]*')
    TOKENS = re.compile(r"\w+|[^\w\s]")

    _LAST_WORD = re.compile(r'(?:\b|\d)([a-zа-я]+)\.$', re.IGNORECASE)
    _FIRST_WORD = re.compile(r'^\W*(\w+)')
    _ENDS_WITH_ONE_LETTER_LAT_AND_DOT = re.compile(r'(\d|\W|\b)([a-zA-Z])\.$')
    _HAS_DOT_INSIDE = re.compile(r'[\w]+\.[\w]+\.$', re.IGNORECASE)
    _LOWER_DOT_UPPER = re.compile(r'\b[a-zа-я]+\.[A-ZА-Я]')
    _INITIALS = re.compile(r'(\W|\b)([A-ZА-Я]{1})\.$')
    _ONLY_RUS_CONSONANTS = re.compile(r'^[бвгджзйклмнпрстфхцчшщ]{1,4}$', re.IGNORECASE)
    _STARTS_WITH_EMPTYNESS = re.compile(r'^\s+')
    _ENDS_WITH_EMOTION = re.compile(r'[!?…]|\.{2,}\s?[)"«»,“]?$')
    _STARTS_WITH_LOWER = re.compile(r'^\s*[–-—-("«]?\s*[a-zа-я]')
    _STARTS_WITH_DIGIT = re.compile(r'^\s*\d')
    _NUMERATION = re.compile(r'^\W*[IVXMCL\d]+\.$')
    _PAIRED_SHORTENING_IN_THE_END = re.compile(r'\b(\w+)\. (\w+)\.\W*$')

    _JOIN = 0
    _MAYBE = 1
    _SPLIT = 2

    JOINING_SHORTENINGS = \
        {'mr', 'mrs', 'ms', 'dr', 'vs', 'англ', 'итал', 'греч', 'евр', 'араб', 'яп', 'слав', 'кит',
         'тел', 'св', 'ул', 'устар', 'им', 'г', 'см', 'д', 'стр', 'корп', 'пл', 'пер', 'сокр', 'рис'}
    SHORTENINGS = \
        {'co', 'corp', 'inc', 'ltd', 'авт', 'адм', 'барр', 'внутр', 'га', 'дифф', 'дол', 'долл', 'зав', 'зам',
         'искл', 'коп', 'корп', 'куб', 'лат', 'мин', 'о', 'обл', 'обр', 'прим', 'проц', 'р', 'ред', 'руб', 'рус',
         'русск', 'сан', 'сек', 'тыс', 'эт', 'яз', 'гос', 'мн', 'жен', 'муж', 'накл', 'повел', 'букв', 'шутл', 'ед'}

    PAIRED_SHORTENINGS = {('и', 'о'), ('т', 'е'), ('т', 'п'), ('у', 'е'), ('н', 'э')}

    @classmethod
    def _regex_split_separators(cls, text: str) -> [str]:
        return [x.strip() for x in cls.SENT_RE.findall(text)]

    @classmethod
    def _is_sentence_end(cls,
                         left: str,
                         right: str) -> int:
        if not cls._STARTS_WITH_EMPTYNESS.match(right):
            return cls._JOIN

        if cls._HAS_DOT_INSIDE.search(left):
            return cls._JOIN

        left_last_word = cls._LAST_WORD.search(left)
        lw = ' '
        if left_last_word:
            lw = left_last_word.group(1)

            if lw.lower() in cls.JOINING_SHORTENINGS:
                return cls._JOIN

            if cls._ONLY_RUS_CONSONANTS.search(lw) and lw[-1].islower():
                return cls._MAYBE

        pse = cls._PAIRED_SHORTENING_IN_THE_END.search(left)
        if pse:
            s1, s2 = pse.groups()
            if (s1, s2) in cls.PAIRED_SHORTENINGS:
                return cls._MAYBE

        right_first_word = cls._FIRST_WORD.match(right)
        if right_first_word:
            rw = right_first_word.group(1)
            if (lw, rw) in cls.PAIRED_SHORTENINGS:
                return cls._MAYBE

        if cls._ENDS_WITH_EMOTION.search(left) and cls._STARTS_WITH_LOWER.match(right):
            return cls._JOIN

        initials = cls._INITIALS.search(left)
        if initials:
            border, _ = initials.groups()
            if (border or ' ') not in "°'":
                return cls._JOIN

        if lw.lower() in cls.SHORTENINGS:
            return cls._MAYBE

        last_letter = cls._ENDS_WITH_ONE_LETTER_LAT_AND_DOT.search(left)
        if last_letter:
            border, _ = last_letter.groups()
            if (border or ' ') not in "°'":
                return cls._MAYBE
        if cls._NUMERATION.match(left):
            return cls._JOIN
        return cls._SPLIT

    @classmethod
    def sent_tokenize(cls, text: str) -> List[str]:
        sentences = []
        sents = cls._regex_split_separators(text)
        si = 0
        processed_index = 0
        sent_start = 0
        while si < len(sents):
            s = sents[si]
            span_start = text[processed_index:].index(s) + processed_index
            span_end = span_start + len(s)
            processed_index += len(s)

            si += 1

            send = cls._is_sentence_end(text[sent_start: span_end], text[span_end:])
            if send == cls._JOIN:
                continue

            if send == cls._MAYBE:
                if cls._STARTS_WITH_LOWER.match(text[span_end:]):
                    continue
                if cls._STARTS_WITH_DIGIT.match(text[span_end:]):
                    continue

            # if not text[sent_start: span_end].strip():
            #     logger.warning("Something went wrong while tokenizing")
            # sentences.append(text[sent_start: span_end].strip())
            sentences += cls._fix(text[sent_start: span_end].strip())
            sent_start = span_end
            processed_index = span_end

        if sent_start != len(text):
            if text[sent_start:].strip():
                # sentences.append(text[sent_start:].strip())
                sentences += cls._fix(text[sent_start:].strip())
        return sentences

    @classmethod
    def _fix(cls, sent):
        pivots = []
        for m in cls._LOWER_DOT_UPPER.finditer(sent):
            start, end = m.span()
            i = m.group().index(".")
            pivots.append(start + i + 1)
        pivots = [0] + pivots + [len(sent)]
        chunks = [sent[i:j] for i, j in zip(pivots[:-1], pivots[1:])]
        return chunks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--corpora_dir")
    parser.add_argument("--output_file")
    parser.add_argument("--num_processes", type=int, default=1)
    args = parser.parse_args()

    print("loading raw texts...")
    texts = set()
    for file in os.listdir(args.corpora_dir):
        with open(os.path.join(args.corpora_dir, file)) as f:
            for i, text in tqdm(enumerate(f)):
                # if i == 1000:
                #     break
                texts.add(text.strip())
    print("corpus size:", len(texts))
    texts = list(texts)
    random.seed(228)
    random.shuffle(texts)

    # print("sentence tokenizing...")
    # res = []
    # batch_size = 500000
    # with Pool(args.num_processes) as p:
    #     for start in range(0, len(texts), batch_size):
    #         print(start)
    #         end = start + batch_size
    #         res += p.map(SentTokenizer.sent_tokenize, texts[start:end])
    #
    # print("saving...")
    # with open(args.output_file, "w") as f:
    #     for sentences in tqdm(res):
    #         for sent in sentences:
    #             f.write(sent + "\n")
    #         f.write("\n")

    with open(args.output_file, "w") as f:
        for text in tqdm(texts):
            for sent in SentTokenizer.sent_tokenize(text):
                f.write(sent + "\n")
            f.write("\n")
