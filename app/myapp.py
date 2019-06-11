import json
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div
from bokeh.transform import factor_cmap
from bokeh.models.widgets import AutocompleteInput, Button
from bokeh.layouts import column, row, widgetbox


###################################################
# PREPARING DATA
###################################################

index = 0
id_sentence = list()
id_word = list()
x = list()
y = list()
sentences = list()
language_sentence = list()
language_word = list()
words = list()
x_words = list()
y_words = list()
links = dict()
with open('data/data_mapping_sentences_words.json', 'r') as file:
    array = json.load(file)
    data = array['content']
    for idx, translations in data.items():
        nbr_lang = len(translations)
        for lang, info in translations.items():
            id_sentence.append(idx)
            language_sentence.append(lang)
            sentences.append(info['sentence'])
            x.append(info['embedding'][0])
            y.append(info['embedding'][1])
            for word, emb in info['words'].items():
                id_word.append(idx)
                language_word.append(lang)
                words.append(word)
                x_words.append(emb[0])
                y_words.append(emb[1])
        for i in range(nbr_lang):
            links[index + i] = [index + j for j in range(nbr_lang) if j != i]
        index += nbr_lang

    source_sentence_dict = {
        "id": id_sentence,
        "x": x,
        "y": y,
        "lang": language_sentence,
        "sentence": sentences
    }

    source_word_dict = {
        "id": id_word,
        "x": x_words,
        "y": y_words,
        "lang": language_word,
        "word": words
    }

source_sentences = ColumnDataSource(source_sentence_dict)
source_words = ColumnDataSource(source_word_dict)
source_words_visible = ColumnDataSource(source_word_dict)

###################################################
# SET UP SENTENCE FIGURE
###################################################

tools = "hover, tap, box_select, reset, help"
p = figure(plot_width=800, plot_height=700, tools=tools, title='Intermediate representations')

cr = p.circle('x', 'y',
              size=6, alpha=0.6,
              hover_color='yellow', hover_alpha=1.0,
              source=source_sentences,
              color=factor_cmap('lang',
                                palette=['red', 'blue', 'green'],
                                factors=['en', 'fr', 'es']),
              legend='lang',
              name="sentences")

p.legend.background_fill_alpha = 0.4

###################################################
# SET UP LINKS BETWEEN TRANSLATIONS
###################################################

source = ColumnDataSource({'x0': [], 'y0': [], 'x1': [], 'y1': []})
sr = p.segment(x0='x0', y0='y0', x1='x1', y1='y1', color='black', alpha=0.6, line_width=1, source=source)

code = """
var links = %s;
var data = {'x0': [], 'y0': [], 'x1': [], 'y1': []};
var cdata = circle.data;
var indices = cb_data.index['1d'].indices;
for (var i = 0; i < indices.length; i++) {
    var ind0 = indices[i]
    for (var j = 0; j < links[ind0].length; j++) {
        var ind1 = links[ind0][j];
        data['x0'].push(cdata.x[ind0]);
        data['y0'].push(cdata.y[ind0]);
        data['x1'].push(cdata.x[ind1]);
        data['y1'].push(cdata.y[ind1]);
    }
}
segment.data = data;
""" % links

###################################################
# ADD LINKS IN HOVERTOOL
###################################################

callback = CustomJS(args={'circle': cr.data_source,
                          'segment': sr.data_source}, code=code)
p.add_tools(HoverTool(callback=callback, renderers=[cr]))
hover = p.select(dict(type=HoverTool))
hover.tooltips = [
    ("sentence", "@sentence"),
]
hover.names = [
    "sentences"
]

###################################################
# SET UP WORD FIGURE
###################################################

p_w = figure(title='Words representations', tools="reset")
p_w.circle('x', 'y',
           size=6, alpha=0.6,
           hover_color='yellow', hover_alpha=1.0,
           selection_color='yellow',
           nonselection_color='white',
           source=source_words_visible,
           color=factor_cmap('lang',
                             palette=['red', 'blue', 'green'],
                             factors=['en', 'fr', 'es']),
           legend='lang',
           name="words")

p_w.legend.background_fill_alpha = 0.4

hover_word = HoverTool(tooltips=[('word', '@word')],
                       names=['words'])
p_w.add_tools(hover_word)


def update_words(attr, old, new):
    source_words_visible_dict = {
        "id": [],
        "x": [],
        "y": [],
        "lang": [],
        "word": []
    }
    for indice in new:
        indice_sentence = source_sentences.data['id'][indice]
        for idx, indice_word in enumerate(source_words.data['id']):
            if indice_word == indice_sentence:
                source_words_visible_dict['id'].append(source_words.data['id'][idx])
                source_words_visible_dict['x'].append(source_words.data['x'][idx])
                source_words_visible_dict['y'].append(source_words.data['y'][idx])
                source_words_visible_dict['lang'].append(source_words.data['lang'][idx])
                source_words_visible_dict['word'].append(source_words.data['word'][idx])
    source_words_visible.data = source_words_visible_dict


source_sentences.selected.on_change('indices', update_words)

text_input = AutocompleteInput(title='Search a sentence', completions=sentences)

reset_button = Button(label='ALL WORDS', button_type='primary')
reset_button.js_on_click(CustomJS(args=dict(source=source_words_visible, words=source_words, p=p), code="""
    p.reset.emit()
    source.data = words.data
"""))

###################################################
# CREATION OF THE LAYOUT
###################################################

window = row(p,
             column(row(widgetbox(text_input), column(Div(text="", height=0), reset_button)),
                    p_w))

curdoc().add_root(window)
