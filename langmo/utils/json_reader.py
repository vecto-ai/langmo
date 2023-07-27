# TODO: consider moving this to kapral
import json
import string

from kapral.corpus import BufferCorpus
from kapral.utils.data import detect_archive_format_and_open
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException


class DocFromJSONFileIter:

    def __init__(self, path):
        self.path = path
        self._gen = self.gen()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def gen(self):
        with detect_archive_format_and_open(self.path) as stream:
            for line in stream:
                yield BufferCorpus(json.loads(line)["text"])


# class JSONLDocIter:
#     def __init__(self, path):
#         self.file_iter = DirIterator(path)
#         self._gen = self.gen()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return next(self._gen)

#     def gen(self):
#         for file_name in self.file_iter:
#             print("processing", file_name)
#             with detect_archive_format_and_open(file_name) as stream:
#                 for line in stream:
#                     yield json.loads(line)["text"]


class QualityFilter:
    def __init__(self):
        self.chars_good = set(string.ascii_letters + " !?,.\"'")

    def text_is_english(self, text) -> bool:
        try:
            langs = detect_langs(text[:60])
        except LangDetectException:
            return False
        lang = langs[0]
        if lang.lang != 'en' or lang.prob < 0.9:
            return False
        return True

    def has_bad_chars(self, p):
        if sum(a in string.ascii_letters for a in p[:5]) < 3:
            return True
        if not p[0] in string.ascii_letters:
            return True
        cnt_ascii = sum(a in self.chars_good for a in p) / len(p)
        if cnt_ascii < 0.9:
            return True
        return False

    def has_boilerplate(self, text):
        if text.startswith("Discussion in"):
            return True
        return False

    def is_good(self, text):
        if self.has_bad_chars(text):
            return False
        if self.has_boilerplate(text):
            return False
        if not self.text_is_english(text):
            return False
        return True

    def get_document_iterator(self, doc_iter):
        for doc in doc_iter:
            if self.is_good(doc):
                doc = doc.replace("\n", " ")
                yield BufferCorpus(doc)
