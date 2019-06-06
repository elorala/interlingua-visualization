import json
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.transform import factor_cmap
from bokeh.models.widgets import AutocompleteInput
from bokeh.layouts import column, row


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

p_w = figure(title='Words', tools="reset")
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

hover_word = HoverTool(tooltips=[('word', '@word')],
                       names=['words'])
p_w.add_tools(hover_word)

source_sentences.selected.js_on_change('indices', CustomJS(args=dict(sentences=source_sentences,
                                                                     words=source_words,
                                                                     words_visible=source_words_visible),
                                                           code="""
        var inds = cb_obj.indices;
        var sentences = sentences.data;
        var words = words.data;
        var words_visible = words_visible.data

        words_visible.id = []
        words_visible.x = []
        words_visible.y = []
        words_visible.lang = []
        words_visible.word = []

        console.log(sentences.id[inds])
        for (var i = 0; i < words.x.length; i++) {
            if (words.id[i] == sentences.id[inds]) {
                words_visible.id.push(words.id[i])
                words_visible.x.push(words.x[i])
                words_visible.y.push(words.y[i])
                words_visible.lang.push(words.lang[i])
                words_visible.word.push(words.word[i])
            }
        }
        words_visible.change.emit();
    """)
                                       )

text_input = AutocompleteInput(title='Search a sentence', completions=sentences)

###################################################
# CREATION OF THE LAYOUT
###################################################

window = row(p, column(p_w, text_input))

curdoc().add_root(window)
