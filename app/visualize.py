import json
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, Slider, TapTool, TextInput
from bokeh.layouts import gridplot, column, layout, row
from bokeh.transform import factor_cmap
from bokeh.palettes import viridis
from bokeh.models.widgets import DataTable, TableColumn
from scipy.spatial.distance import cdist


def plot_extraction(lang, input_file):
    x = list()
    y = list()
    sentences = list()
    with open(input_file, 'r') as file:
        array = json.load(file)
        data = array['content']
        for translation in data:
            x.append(translation['weights_{}'.format(lang)][0])
            y.append(translation['weights_{}'.format(lang)][1])
            sentences.append(translation[lang])

            info_dict = {
                "X": x,
                "Y": y,
                "sentence": sentences,
            }
    return ColumnDataSource(info_dict)


def full_lang_plot():
    output_file("dashboards/gridplot.html")

    tooltips = [
        ("(x,y)", "($x, $y)"),
        ("sentences", "$sentences"),
    ]

    full_p = figure(title="3 Languages", tooltips=tooltips, plot_width=500, plot_height=500)
    full_p.circle('X', 'Y', size=10, color="navy", alpha=0.5,
                  source=plot_extraction('f1', 'data_umapi_l1.json'), legend='EN')
    full_p.circle('X', 'Y', size=10, color="red", alpha=0.5,
                  source=plot_extraction('f2', 'data_umapi_l1.json'), legend='FR')
    full_p.circle('X', 'Y', size=10, color="green", alpha=0.5,
                  source=plot_extraction('f3', 'data_umapi_l1.json'), legend='ES')

    en_fr_p = figure(title="EN/FR", plot_width=500, plot_height=500)
    en_fr_p.circle('X', 'Y', size=10, color="navy", alpha=0.5,
                   source=plot_extraction('f1', 'data_umapi_l1.json'), legend='EN')
    en_fr_p.circle('X', 'Y', size=10, color="red", alpha=0.5,
                   source=plot_extraction('f2', 'data_umapi_l1.json'), legend='FR')

    en_es_p = figure(title="EN/ES", plot_width=500, plot_height=500)
    en_es_p.circle('X', 'Y', size=10, color="navy", alpha=0.5,
                   source=plot_extraction('f1', 'data_umapi_l1.json'), legend='EN')
    en_es_p.circle('X', 'Y', size=10, color="green", alpha=0.5,
                   source=plot_extraction('f3', 'data_umapi_l1.json'), legend='ES')

    es_fr_p = figure(title="ES/FR", plot_width=500, plot_height=500)
    es_fr_p.circle('X', 'Y', size=10, color="red", alpha=0.5,
                   source=plot_extraction('f2', 'data_umapi_l1.json'), legend='FR')
    es_fr_p.circle('X', 'Y', size=10, color="green", alpha=0.5,
                   source=plot_extraction('f3', 'data_umapi_l1.json'), legend='ES')

    grid = gridplot([[full_p, en_fr_p], [en_es_p, es_fr_p]])

    show(grid)


def compare_umap_plots():
    output_file("dashboards/compare_umap.html", mode="inline")

    tools = "pan,wheel_zoom,box_zoom,reset"
    tooltips = [
        ("sentences", "$sentences"),
    ]

    en_fr_p = figure(title="EN/FR", plot_width=500, plot_height=500, tools=tools, tooltips=tooltips)
    en_fr_p.circle('X', 'Y', size=10, color="navy", alpha=0.5,
                   source=plot_extraction('f2', 'umap_fr_en.json'), legend='EN')
    en_fr_p.circle('X', 'Y', size=10, color="red", alpha=0.5,
                   source=plot_extraction('f1', 'umap_fr_en.json'), legend='FR')

    en_es_p = figure(title="EN/ES", plot_width=500, plot_height=500, x_range=en_fr_p.x_range, y_range=en_fr_p.y_range)
    en_es_p.circle('X', 'Y', size=10, color="navy", alpha=0.5,
                   source=plot_extraction('f1', 'umap_en_es.json'), legend='EN')
    en_es_p.circle('X', 'Y', size=10, color="green", alpha=0.5,
                   source=plot_extraction('f2', 'umap_en_es.json'), legend='ES')

    es_fr_p = figure(title="ES/FR", plot_width=500, plot_height=500, x_range=en_fr_p.x_range, y_range=en_fr_p.y_range)
    es_fr_p.circle('X', 'Y', size=10, color="red", alpha=0.5,
                   source=plot_extraction('f1', 'umap_fr_es.json'), legend='FR')
    es_fr_p.circle('X', 'Y', size=10, color="green", alpha=0.5,
                   source=plot_extraction('f2', 'umap_fr_es.json'), legend='ES')

    grid = gridplot([[en_fr_p, en_es_p], [es_fr_p, None]])

    show(grid)


def highlight_sentences(input_file):
    output_file("dashboards/highlight.html")
    index = 0
    x = list()
    y = list()
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
            links[index] = [index + 1, index + 2]
            links[index + 1] = [index, index + 2]
            links[index + 2] = [index, index + 1]
            index += 3

        source_dict = {
            "x": x,
            "y": y,
            "lang": language,
            "sentence": sentences
        }

    source = ColumnDataSource(source_dict)

    p = figure(title='word embeddings', plot_width=1000, plot_height=700)
    p.scatter('x', 'y', source=source, size=10, alpha=0.4,
              color=factor_cmap('lang', ['red', 'blue', 'green'], ['f1', 'f2', 'f3']),
              legend='lang')
    show(p)
    # view1 = CDSView(source=source, filters=[GroupFilter(column_name='lang', group='f1')])
    # view2 = CDSView(source=source, filters=[GroupFilter(column_name='lang', group='f2')])
    # view3 = CDSView(source=source, filters=[GroupFilter(column_name='lang', group='f3')])
    # tools = "hover,box_select,lasso_select,help"
    #
    # full_p = figure(title="3 Languages", tools=tools, plot_width=1000, plot_height=700)
    # full_p.circle('x', 'y', size=10, color="navy", alpha=0.5,
    #               source=source, view=view1, legend='EN')
    # full_p.circle('x', 'y', size=10, color="red", alpha=0.5,
    #               source=source, view=view2, legend='FR')
    # full_p.circle('x', 'y', size=10, color="green", alpha=0.5,
    #               source=source, view=view3, legend='ES')
    #
    # hover = full_p.select(dict(type=HoverTool))
    # hover.tooltips = [
    #     ("(x,y)", "(@x, @y)"),
    #     ("sentences", "@sentence"),
    # ]
    #
    # show(full_p)


def principal_window(input_file):
    output_file(filename="dashboards/hover_callback.html", title="Intermediate representations")

    ###################################################
    # PREPARING DATA
    ###################################################

    index = 0
    x = list()
    y = list()
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
            links[index] = [index + 1, index + 2]
            links[index + 1] = [index, index + 2]
            links[index + 2] = [index, index + 1]
            index += 3

        source_dict = {
            "x": x,
            "y": y,
            "lang": language,
            "sentence": sentences
        }

    source_circle = ColumnDataSource(source_dict)

    ###################################################
    # SET UP MAIN FIGURE
    ###################################################

    tools = "hover, tap, box_zoom, box_select, reset, help"
    p = figure(plot_width=800, plot_height=700, tools=tools, title='Intermediate representations')

    cr = p.circle('x', 'y',
                  size=10, alpha=0.4,
                  hover_color='yellow', hover_alpha=1.0,
                  source=source_circle,
                  color=factor_cmap('lang', ['red', 'blue', 'green'], ['en', 'fr', 'es']),
                  legend='lang',
                  name="data")

    p.legend.click_policy = "hide"

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
        "data"
    ]

    ###################################################
    # ADD TRANSLATIONS SECTION
    ###################################################

    div = Div(height=700)

    ###################################################
    # ADD TAP EVENT
    ###################################################

    taptool = p.select(type=TapTool)
    taptool.callback = CustomJS(args=dict(div=div, data=source_circle, link=source), code="""
        var data = data.data;
        var lines = data['sentence'];
        div.text = lines.join("\\n");
    """)

    ###################################################
    # CREATION OF THE LAYOUT
    ###################################################

    window = layout([p], sizing_mode='stretch_both')
    # curdoc().add_root(window)
    show(window)


def comparing_layers(input_files, nbr_lang):
    output_file(filename="../dashboards/intermediate_layer_%s.html" % nbr_lang, title='Decoding Layers')

    ###################################################
    # PREPARING DATA
    ###################################################
    nbr_layer = len(input_files)
    sources = []
    sources_seg = []
    pourcentages = list()
    for input_file in input_files:
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
                if nbr_lang == 3:
                    links[index] = [index + 1, index + 2]
                    links[index + 1] = [index, index + 2]
                    links[index + 2] = [index, index + 1]
                    index += 3
                else:
                    links[index] = [index + 1]
                    links[index + 1] = [index]
                    index += 2

            source_dict = {
                "x": x,
                "y": y,
                "lang": language,  # enes or eses OR esen enen fren
                "sentence": sentences
            }

            vectors = list(zip(source_dict["x"], source_dict["y"]))
            total = len(vectors)
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

            vect = len(dist)
            pourcentage = (vect / total) * 100

        pourcentages.append(round(pourcentage, 3))

        source_seg = ColumnDataSource({
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'dist': dist})

        sources_seg.append(source_seg)
        source_circle = ColumnDataSource(source_dict)
        sources.append(source_circle)

    ###################################################
    # SET UP MAIN FIGURE
    ###################################################

    tools = "save"

    figures = []
    for layer in range(nbr_layer):
        p = figure(plot_width=400, plot_height=400, tools=tools, title='LAYER {}'.format(layer))

        if nbr_lang == 2:
            cr = p.circle('x', 'y',
                          size=5, alpha=0.5,
                          hover_color='yellow', hover_alpha=1.0,
                          source=sources[layer],
                          color=factor_cmap('lang', ['red', 'blue'], ['enes', 'eses']),
                          legend='lang',
                          name='data')
        else:
            cr = p.circle('x', 'y',
                          size=5, alpha=0.5,
                          hover_color='yellow', hover_alpha=1.0,
                          source=sources[layer],
                          color=factor_cmap('lang', ['red', 'blue', 'green'], ['enen', 'esen', 'fren']),
                          legend='lang',
                          name='data')

        ###################################################
        # SET UP LEGEND SPECIFICATION
        ###################################################
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.4

        ###################################################
        # SET UP LINKS BETWEEN TRANSLATIONS
        ###################################################
        # source = ColumnDataSource({'x0': [], 'y0': [], 'x1': [], 'y1': []})
        sr = p.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                       color='black',
                       alpha=0.6,
                       line_width=0.3,
                       source=sources_seg[layer],
                       name='seg')

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

        # callback = CustomJS(args={'circle': cr.data_source,
        #                           'segment': sr.data_source}, code=code)
        # p.add_tools(HoverTool(callback=callback, renderers=[cr]))
        # hover = p.select(dict(type=HoverTool))
        # hover.tooltips = [
        #     ("sentences", "@sentence"),
        # ]
        # hover.names = [
        #     "data"
        # ]
        hover1 = HoverTool(tooltips=[("sentences", "@sentence")], names=['data'], renderers=[cr])
        hover2 = HoverTool(tooltips=[("distance", "@dist")], names=['seg'], renderers=[sr])
        p.add_tools(hover1)
        p.add_tools(hover2)

        figures.append(p)

        # hover2 = p.select(dict(type=HoverTool))
        # hover2.tooltips = [
        #     ("distance", "$dist"),
        # ]
        # hover2.names = [
        #     "seg"
        # ]
    grid = gridplot([figures[:3], figures[3:]])
    show(grid)


def principal_window_without_sentences(input_file):
    output_file("../dashboards/without_sentence.html")

    ###################################################
    # PREPARING DATA
    ###################################################

    index = 0
    x = list()
    y = list()
    language = list()
    links = dict()
    with open(input_file, 'r') as file:
        array = json.load(file)
        data = array['content']
        for translation in data:
            for lang, info in translation.items():
                language.append(lang)
                x.append(info[0])
                y.append(info[1])
            links[index] = [index + 1, index + 2, index + 3]
            links[index + 1] = [index, index + 2, index + 3]
            links[index + 2] = [index, index + 1, index + 3]
            links[index + 3] = [index, index + 1, index + 2]
            index += 4

        source_dict = {
            "x": x,
            "y": y,
            "lang": language,
        }

    source_circle = ColumnDataSource(source_dict)

    ###################################################
    # SET UP MAIN FIGURE
    ###################################################

    tools = "tap, box_zoom, box_select, reset, save, help"
    p = figure(plot_width=800, plot_height=700, tools=tools,
               title='INTERMEDIATE REPRESENTATIONS FOR INTERLINGUA WITHOUT DISTANCE')

    cr = p.circle('x', 'y',
                  size=5, alpha=0.5,
                  hover_color='yellow', hover_alpha=1.0,
                  source=source_circle,
                  color=factor_cmap('lang', ['red', 'blue', 'green', 'orange'], ['en', 'fr', 'es', 'de']),
                  legend='lang',
                  name="data")

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
    p.add_tools(HoverTool(callback=callback, renderers=[cr], tooltips=None))

    ###################################################
    # CREATION OF THE LAYOUT
    ###################################################

    window = layout([[p]], sizing_mode='stretch_both')
    # curdoc().add_root(window)
    show(window)


def principal_window_words(inputs_file):
    output_file(filename="../dashboards/words_representations.html", title="Words representations")

    ###################################################
    # PREPARING DATA
    ###################################################
    sources = []
    sources_visible = []
    languages = []
    colors = viridis(len(inputs_file))
    for input_file in inputs_file:
        x = list()
        y = list()
        words = list()
        language = input_file[11:-5]
        languages.append(language)
        with open(input_file, 'r') as file:
            array = json.load(file)
            data = array['content']
            for translation in data:
                for word, vector in translation.items():
                    words.append(word)
                    x.append(vector[0])
                    y.append(vector[1])

            source_dict = {
                'x': x,
                'y': y,
                'words': words,
            }
            source = ColumnDataSource(source_dict)
            source_visible = ColumnDataSource({
                'x': x[:len(x) // 100 * 10],
                'y': y[:len(y) // 100 * 10],
                'words': words[:len(words) // 100 * 10],
            })
        sources.append(source)
        sources_visible.append(source_visible)

    ###################################################
    # SET UP MAIN FIGURE
    ###################################################

    tools = "hover, tap, box_zoom, box_select, reset, help"
    p = figure(tools=tools, title='Words intermediate representations\n', plot_width=1000, plot_height=650)

    for index in range(len(sources_visible)):
        p.circle('x', 'y',
                 size=4, alpha=0.4,
                 hover_color='red', hover_alpha=1.0,
                 selection_color='red',
                 nonselection_color='white',
                 source=sources_visible[index],
                 color=colors[index],
                 legend=languages[index],
                 name="data_{}".format(index))

    p.legend.click_policy = "hide"

    ###################################################
    # ADD LINKS IN HOVERTOOL
    ###################################################

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("words", "@words"),
    ]

    ###################################################
    # SET UP SLIDER
    ###################################################

    slider = Slider(title='Percentage of words',
                    value=10,
                    start=1,
                    end=100,
                    step=1)
    slider.callback = CustomJS(args=dict(sources_visible=sources_visible, sources=sources), code="""
            var percentage = cb_obj.value;
            // Get the data from the data sources
            for(var i=0; i < sources.length; i++) {
                var point_visible = sources_visible[i].data;
                var point_available = sources[i].data;
                var nbr_points = (point_available.x.length / 100) * percentage
    
                point_visible.x = []
                point_visible.y = []
    
                // Update the visible data
                for(var j = 0; j < nbr_points; j++) {  
                    point_visible.x.push(point_available.x[j]);
                    point_visible.y.push(point_available.y[j]);
                }  
                sources_visible[i].change.emit();
            }
            """)

    ###################################################
    # SET UP DATATABLE
    ###################################################

    columns0 = [TableColumn(field="words", title="Words in English")]
    columns1 = [TableColumn(field="words", title="Words in Spanish")]
    columns2 = [TableColumn(field="words", title="Words in French")]

    data_table0 = DataTable(source=sources_visible[0], columns=columns0, width=300, height=175)
    data_table1 = DataTable(source=sources_visible[1], columns=columns1, width=300, height=175)
    data_table2 = DataTable(source=sources_visible[2], columns=columns2, width=300, height=175)

    ###################################################
    # CREATION OF THE LAYOUT
    ###################################################

    window = layout([[p, column(slider, data_table0, data_table1, data_table2)]])
    # curdoc().add_root(window)
    show(window)


def mapping_sentences_words(input_file):
    output_file(filename="../dashboards/sentences_words.html", title="Intermediate representations")

    ###################################################
    # PREPARING DATA
    ###################################################

    index = 0
    x = list()
    y = list()
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
            links[index] = [index + 1, index + 2]
            links[index + 1] = [index, index + 2]
            links[index + 2] = [index, index + 1]
            index += 3

        source_dict = {
            "x": x,
            "y": y,
            "lang": language,
            "sentence": sentences
        }

    source_sentences = ColumnDataSource(source_dict)

    ###################################################
    # SET UP SENTENCE FIGURE
    ###################################################

    tools = "hover, tap, box_select, reset, help"
    p = figure(plot_width=800, plot_height=700, tools=tools, title='Intermediate representations')

    cr = p.circle('x', 'y',
                  size=6, alpha=0.4,
                  hover_color='yellow', hover_alpha=1.0,
                  source=source_sentences,
                  color=factor_cmap('lang', ['red', 'blue', 'green'], ['en', 'fr', 'es']),
                  legend='lang',
                  name="data")

    p.legend.click_policy = "hide"

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
        "data"
    ]

    ###################################################
    # SET UP WORD FIGURE
    ###################################################

    source_words = ColumnDataSource(data=dict(x=[], y=[]))
    p_w = figure(title='Words', tools="")
    p_w.circle('x', 'y', source=source_words, alpha=0.6)

    source_sentences.selected.js_on_change('indices', CustomJS(args=dict(s1=source_sentences, s2=source_words), code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            d2['x'] = []
            d2['y'] = []
            for (var i = 0; i < inds.length; i++) {
                d2['x'].push(d1['x'][inds[i]])
                d2['y'].push(d1['y'][inds[i]])
            }
            s2.change.emit();
        """)
                                           )

    ###################################################
    # CREATION OF THE LAYOUT
    ###################################################

    # window = layout([p, words], sizing_mode='stretch_both')
    window = row(p, p_w)
    show(window)


if __name__ == '__main__':
    # full_lang_plot()
    # compare_umap_plots()
    # highlight_sentences("data_umapi_l1.json")
    # principal_window("data_umap_3lang.json")
    # principal_window_without_sentences('embeddings/umap.json')
    # comparing_layers(['decodings_es_layer0.json',
    #                   'decodings_es_layer1.json',
    #                   'decodings_es_layer2.json',
    #                   'decodings_es_layer3.json',
    #                   'decodings_es_layer4.json',
    #                   'decodings_es_layer5.json'], 2)
    # comparing_layers(['decodings_en_layer0.json',
    #                   'decodings_en_layer1.json',
    #                   'decodings_en_layer2.json',
    #                   'decodings_en_layer3.json',
    #                   'decodings_en_layer4.json',
    #                   'decodings_en_layer5.json'], 3)
    principal_window_words(['../data/data_words_en.json',
                            '../data/data_words_es.json',
                            '../data/data_words_fr.json'])
    mapping_sentences_words("../data/data_umap_3lang.json")
