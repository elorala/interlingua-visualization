import csv
import json
import numpy as np
import umap.umap_ as umap
from tqdm import tqdm
import logging
import spacy

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class DataProcessing:
    def __init__(self, lang, raw_embeddings, raw_sentences):
        self.lang = lang
        with open(raw_embeddings) as emb:
            self.embeddings = json.load(emb)
        with open(raw_sentences) as sent:
            self.sentences = sent.read().split('\n')


class DataLayer:
    def __init__(self, lang_in, lang_out, raw_encodings):
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.lang = lang_in+lang_out
        self.raw_file = raw_encodings
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            with open(self.raw_file) as file:
                return json.load(file)

    def load_embeddings(self):
        return self.embeddings


class Sentences:
    def __init__(self, lang, raw_sentences, raw_embeddings):
        self.lang = lang
        self.raw_sentences = raw_sentences
        self.raw_embeddings = raw_embeddings
        self._sentences = None
        self._words = None
        self._embeddings = None
        self.nlp = None
        self.useful_sentences = None
        self.umap_embeddings = None

    def __len__(self):
        return len(self.sentences)

    @property
    def sentences(self):
        if self._sentences is None:
            with open(self.raw_sentences) as sent:
                sentences = sent.read().split('\n')
            return sentences

    @property
    def words(self):
        if self._words is None:
            words = []
            try:
                if self.lang == 'en':
                    self.nlp = spacy.load('en_core_web_sm')
                else:
                    self.nlp = spacy.load('{}_core_news_sm'.format(self.lang))
            except OSError:
                print('NLP model not found')

            self.load_sentences()
            logger.info('Loading words for each sentence')
            for sentence in self.sentences:
                sentence_words = []
                doc = self.nlp(sentence)
                for token in doc:
                    sentence_words.append(token.text)
                words.append(sentence_words)
            return words

    @property
    def embeddings(self):
        if self._embeddings is None:
            with open(self.raw_embeddings) as emb:
                embeddings = json.load(emb)
            return embeddings

    def load_sentences(self):
        return self.sentences

    def load_words(self):
        return self.words

    def load_umap_embeddings(self):
        pass


class WordEmbeddings:
    def __init__(self, lang, raw_embeddings):
        self.lang = lang
        self.raw_embeddings = raw_embeddings
        self.full_data = None
        self.words = list()
        self.embeddings = list()
        self.umap_embeddings = None

    def load_data(self):
        if self.full_data is None:
            with open(self.raw_embeddings) as emb:
                logger.info('File {} open'.format(self.raw_embeddings))
                self.full_data = json.load(emb)
            for word, embedding in self.full_data.items():
                logger.info('Adding {}'.format(word))
                self.words.append(word)
                self.embeddings.append(embedding)

    def load_umap_embeddings(self):
        if self.umap_embeddings is None:
            logger.info('Running UMAP for {} words'.format(self.lang))
            self.umap_embeddings = umap.UMAP(n_neighbors=10,
                                             min_dist=0.005,
                                             metric='correlation').fit_transform(self.embeddings)


# INITIAL VERSION
def prepare_input_file(nbr_sentences, data_1, data_2, data_3=None):

    umap_embeddings = []
    umap_sentences = []
    umap_langs = []

    if data_3 is not None:
        limited_data = [k for k in data_1.embeddings.keys()
                        if k in data_2.embeddings.keys() and k in data_3.embeddings.keys()][:nbr_sentences]

        for i in limited_data:
            matrix = data_1.embeddings[i]['encoding']
            f1 = np.asarray(matrix).flatten()

            umap_embeddings.append(f1)
            umap_sentences.append(data_1.sentences[int(i)])
            umap_langs.append(data_1.lang)

            matrix = data_2.embeddings[i]['encoding']
            f2 = np.asarray(matrix).flatten()

            umap_embeddings.append(f2)
            umap_sentences.append(data_2.sentences[int(i)])
            umap_langs.append(data_2.lang)

            matrix = data_3.embeddings[i]['encoding']
            f3 = np.asarray(matrix).flatten()

            umap_embeddings.append(f3)
            umap_sentences.append(data_3.sentences[int(i)])
            umap_langs.append(data_3.lang)

    else:
        limited_data = [k for k in data_1.embeddings.keys()
                        if k in data_2.embeddings.keys()][:nbr_sentences]

        for i in limited_data:
            matrix = data_1.embeddings[i]['encoding']
            f1 = np.asarray(matrix).flatten()

            umap_embeddings.append(f1)
            umap_sentences.append(data_1.sentences[int(i)])
            umap_langs.append(data_1.lang)

            matrix = data_2.embeddings[i]['encoding']
            f2 = np.asarray(matrix).flatten()

            umap_embeddings.append(f2)
            umap_sentences.append(data_2.sentences[int(i)])
            umap_langs.append(data_2.lang)

    return umap_embeddings, umap_sentences, umap_langs


# VERSION WITH EMBEDDINGS AND SENTENCES SEPARATED : COMPARE INTERMEDIATE LAYERS
def prepare_input_file_layer(layer_1, sent_1, layer_2, sent_2, layer_3=None, sent_3=None):

    umap_embeddings = []
    umap_sentences = []
    umap_langs = []

    if layer_3 is not None:
        limited_data = [k for k in layer_1.embeddings.keys()
                        if k in layer_2.embeddings.keys() and k in layer_3.embeddings.keys()]
        logger.debug('Index of embeddings : %s' % limited_data)
        for i in tqdm(limited_data):
            matrix = layer_1.embeddings[i]['encoding']
            f1 = np.asarray(matrix).flatten()

            umap_embeddings.append(f1)
            umap_sentences.append(sent_1.sentences[int(i)])
            umap_langs.append(layer_1.lang)

            matrix = layer_2.embeddings[i]['encoding']
            f2 = np.asarray(matrix).flatten()

            umap_embeddings.append(f2)
            umap_sentences.append(sent_2.sentences[int(i)])
            umap_langs.append(layer_2.lang)

            matrix = layer_3.embeddings[i]['encoding']
            f3 = np.asarray(matrix).flatten()

            umap_embeddings.append(f3)
            umap_sentences.append(sent_3.sentences[int(i)])
            umap_langs.append(layer_3.lang)

    else:
        limited_data = [k for k in layer_1.embeddings.keys()
                        if k in layer_2.embeddings.keys()]
        logger.debug('Index of embeddings : %s' % limited_data)
        for i in tqdm(limited_data):
            matrix = layer_1.embeddings[i]['encoding']
            f1 = np.asarray(matrix).flatten()

            umap_embeddings.append(f1)
            umap_sentences.append(sent_1.sentences[int(i)])
            umap_langs.append(layer_1.lang)

            matrix = layer_2.embeddings[i]['encoding']
            f2 = np.asarray(matrix).flatten()

            umap_embeddings.append(f2)
            umap_sentences.append(sent_2.sentences[int(i)])
            umap_langs.append(layer_2.lang)

    return umap_embeddings, umap_sentences, umap_langs


def umap_reduction(umap_embeddings):
    return umap.UMAP(n_neighbors=10, min_dist=0.005, metric='correlation').fit_transform(umap_embeddings)


def prepare_output_file(nbr_lang, embeddings, umap_sentences, umap_langs, name_output):
    data = dict()
    data['type'] = 'UMAP'
    data['content'] = []

    if nbr_lang == 3:
        for i in tqdm(range(0, len(embeddings), 3)):
            try:
                data['content'].append({umap_langs[i]: [umap_sentences[i], embeddings[i].tolist()],
                                        umap_langs[i+1]: [umap_sentences[i+1], embeddings[i+1].tolist()],
                                        umap_langs[i+2]: [umap_sentences[i+2], embeddings[i+2].tolist()]})
            except Exception as e:
                print('FAILED {}'.format(e))
                failed = failed + 1
    else:
        for i in tqdm(range(0, len(embeddings), 2)):
            try:
                data['content'].append({umap_langs[i]: [umap_sentences[i], embeddings[i].tolist()],
                                        umap_langs[i+1]: [umap_sentences[i+1], embeddings[i+1].tolist()]})
            except Exception as e:
                print('FAILED {}'.format(e))
                failed = failed + 1

    with open(name_output, 'w') as file:
        json.dump(data, file)


def prepare_output_file_words(data_words, name_output):
    data = dict()
    data['content'] = []

    for i in tqdm(range(0, len(data_words.umap_embeddings))):
        try:
            data['content'].append({data_words.words[i]: data_words.umap_embeddings[i].tolist()})
        except Exception as e:
            print('FAILED {}'.format(e))
            failed = failed + 1

    with open(name_output, 'w') as file:
        json.dump(data, file)


def preprocess_data_decoding_layer():
    logger.info('Loading sentences in each language')
    logger.info('Loading english sentences')
    en = Sentences('en', 'encodings/newstest2013.tc.en')
    en.load_sentences()
    logger.info('Loading french sentences')
    fr = Sentences('fr', 'encodings/newstest2013.tc.fr')
    fr.load_sentences()
    logger.info('Loading spanish sentences')
    es = Sentences('es', 'encodings/newstest2013.tc.es')
    es.load_sentences()
    logger.info('Loading german sentences')
    de = Sentences('de', 'encodings/newstest2013.tc.de')
    de.load_sentences()

    logger.info('Loading embeddings for each layer')
    for layer in tqdm(range(5, 6)):
        logger.info('Loading embeddings for the layer %s' % layer)
        logger.info('Loading esen embeddings')
        esen_emb = DataLayer('es', 'en', 'embeddings/decodings-layer%s/decodings-esen-%s.json' % (layer, layer))
        esen_emb.load_embeddings()
        logger.info('Nbr of esen embeddings : %s' % len(esen_emb.embeddings))
        logger.info('Loading enen embeddings')
        enen_emb = DataLayer('en', 'en', 'embeddings/decodings-layer%s/decodings-enen-%s.json' % (layer, layer))
        enen_emb.load_embeddings()
        logger.info('Nbr of enen embeddings : %s' % len(enen_emb.embeddings))
        logger.info('Loading fren embeddings')
        fren_emb = DataLayer('fr', 'en', 'embeddings/decodings-layer%s/decodings-fren-%s.json' % (layer, layer))
        fren_emb.load_embeddings()
        logger.info('Nbr of enen embeddings : %s' % len(fren_emb.embeddings))

        # logger.info('Create input data')
        # embeddings, sentences, langs = prepare_input_file_layer(esen_emb, en, enen_emb, es, fren_emb, fr)

        # logger.info('Running umap reduction')
        # embeddings = umap_reduction(embeddings)
        #
        # logger.info('Create output file')
        # prepare_output_file(nbr_lang=3,
        #                     embeddings=embeddings,
        #                     umap_sentences=sentences,
        #                     umap_langs=langs,
        #                     name_output='decodings_en_layer%s.json' % layer)


def preprocess_words_data():
    en = WordEmbeddings('en', 'embeddings/embedding-en.json')
    en.load_data()
    en.load_umap_embeddings()
    fr = WordEmbeddings('fr', 'embeddings/embedding-fr.json')
    fr.load_data()
    fr.load_umap_embeddings()
    es = WordEmbeddings('es', 'embeddings/embedding-es.json')
    es.load_data()
    es.load_umap_embeddings()

    logger.info('Preparing output file for EN')
    prepare_output_file_words(en, 'data_words_en.json')
    logger.info('Preparing output file for FR')
    prepare_output_file_words(fr, 'data_words_fr.json')
    logger.info('Preparing output file for ES')
    prepare_output_file_words(es, 'data_words_es.json')


# Input order : en / fr / es
def map_sentences_to_words(inputs_words, outputs_words):
    en = Sentences('en', 'encodings/newstest2013.tc.en')
    en.load_words()
    fr = Sentences('fr', 'encodings/newstest2013.tc.fr')
    fr.load_words()
    es = Sentences('es', 'encodings/newstest2013.tc.es')
    es.load_words()

    en_data = dict()
    en_data['content'] = []
    fr_data = dict()
    fr_data['content'] = []
    es_data = dict()
    es_data['content'] = []
    for input_words in inputs_words:
        with open(input_words, 'r') as file:
            in_words = json.load(file)
            for sentence in en.words:
                for word in sentence:
                    if word in in_words:
                        en_data['content'].append()






if __name__ == '__main__':
    map_sentences_to_words()
