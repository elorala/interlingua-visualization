import json
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Select
from bokeh.layouts import column, gridplot
from scipy.spatial.distance import cdist


class Plot:
    def __init__(self, input_files, langs):
        self.input_files = input_files
        self.langs = langs
        self.nbr_lang = len(langs)
        self.sources = list()
        self.sources_dict = list()
        self.sources_seg = list()
        self.sources_seg_dict = list()

    def data_preprocessing(self):
        for input_file in self.input_files:
            index = 0
            x = list()
            y = list()
            x0 = list()
            y0 = list()
            x1 = list()
            y1 = list()
            sentences = list()
            language = list()
            links = dict()
            with open(input_file, 'r') as file:
                array = json.load(file)
                data = array['content']
                for translation in data:
                    for lang, info in translation.items():
                        language.append(lang)
                        sentences.append(info[0])
                        x.append(info[1][0])
                        y.append(info[1][1])
                    for i in range(self.nbr_lang):
                        links[index + i] = [index + j for j in range(self.nbr_lang) if j != i]
                    index += self.nbr_lang
                source_dict = {
                    "x": x,
                    "y": y,
                    "lang": language,  # en>es or es>es OR es>en en>en fr>en
                    "sentence": sentences
                }

                vectors = list(zip(source_dict["x"], source_dict["y"]))
                distances = cdist(vectors, vectors, metric='cosine')
                dist = list()
                for ind, x_coord in enumerate(source_dict["x"]):
                    for ind1 in links.get(ind):
                        if (distances[ind][ind1] > 1.0) and (ind1 > ind):
                            x0.append(x_coord)
                            x1.append(source_dict["x"][ind1])
                            dist.append(distances[ind][ind1])

                for ind, y_coord in enumerate(source_dict["y"]):
                    for ind1 in links.get(ind):
                        if (distances[ind][ind1] > 1.0) and (ind1 > ind):
                            y0.append(y_coord)
                            y1.append(source_dict["y"][ind1])

            source_seg_dict = {
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
                'dist': dist}
            source_seg = ColumnDataSource(source_seg_dict)
            self.sources_seg.append(source_seg)
            self.sources_seg_dict.append(source_seg_dict)

            source_circle = ColumnDataSource(source_dict)
            self.sources.append(source_circle)
            self.sources_dict.append(source_dict)


###################################################
# PREPARING DATA
###################################################
input_files_en = ['decodings/decodings_en_layer0.json',
                  'decodings/decodings_en_layer1.json',
                  'decodings/decodings_en_layer2.json',
                  'decodings/decodings_en_layer3.json',
                  'decodings/decodings_en_layer4.json',
                  'decodings/decodings_en_layer5.json']
input_files_es = ['decodings/decodings_es_layer0.json',
                  'decodings/decodings_es_layer1.json',
                  'decodings/decodings_es_layer2.json',
                  'decodings/decodings_es_layer3.json',
                  'decodings/decodings_es_layer4.json',
                  'decodings/decodings_es_layer5.json']

nbr_layer = len(input_files_en)  # Same number of layers is needed

plot_en = Plot(input_files_en, ['enen', 'esen', 'fren'])
plot_en.data_preprocessing()

plot_es = Plot(input_files_es, ['enes', 'eses'])
plot_es.data_preprocessing()

current_plot = plot_en
###################################################
# SET UP MAIN FIGURE
###################################################
tools = "pan, save"
colors = ['red', 'green', 'blue']

figures = []
cr = []
sr = []
for layer in range(nbr_layer):
    p = figure(plot_width=450, plot_height=450, tools=tools, title='LAYER {}'.format(layer))
    cr.append(p.circle('x', 'y',
                       size=5, alpha=0.6,
                       hover_color='yellow', hover_alpha=1.0,
                       source=plot_en.sources[layer],
                       color=factor_cmap('lang',
                                         palette=colors[:current_plot.nbr_lang],
                                         factors=current_plot.langs),
                       legend='lang',
                       name='decoder_{}'.format(layer)))

    ###################################################
    # SET UP LEGEND SPECIFICATION
    ###################################################
    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.4

    ###################################################
    # SET UP LINKS BETWEEN TRANSLATIONS
    ###################################################
    sr.append(p.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                        color='black',
                        alpha=0.6,
                        line_width=0.3,
                        source=current_plot.sources_seg[layer],
                        name='seg_{}'.format(layer)))

    hover1 = HoverTool(tooltips=[("sentences", "@sentence")],
                       names=['decoder_{}'.format(layer)],
                       renderers=[cr[layer]])
    hover2 = HoverTool(tooltips=[("distance", "@dist")],
                       names=['seg_{}'.format(layer)],
                       renderers=[sr[layer]])
    p.add_tools(hover1)
    p.add_tools(hover2)

    figures.append(p)

###################################################
# SET UP SELECT LANGUAGE
###################################################
select = Select(title="Select a language", options=['decoders to English', 'decoders to Spanish'])


def update_plots(attrname, old, new):
    if new == 'decoders to English':
        new_plot = plot_en
    if new == 'decoders to Spanish':
        new_plot = plot_es

    for new_layer in range(nbr_layer):
        current_plot.sources[new_layer].data = new_plot.sources_dict[new_layer]
        current_plot.sources_seg[new_layer].data = new_plot.sources_seg_dict[new_layer]
        cr[new_layer].glyph.fill_color = factor_cmap('lang',
                                                     palette=colors[:new_plot.nbr_lang],
                                                     factors=new_plot.langs)
        cr[new_layer].glyph.line_color = factor_cmap('lang',
                                                     palette=colors[:new_plot.nbr_lang],
                                                     factors=new_plot.langs)


select.on_change('value', update_plots)

grid = gridplot([figures[:(nbr_layer // 2)], figures[(nbr_layer // 2):]])
board = column(select, grid)
curdoc().title = 'Decoder layers viewer'
curdoc().add_root(board)
