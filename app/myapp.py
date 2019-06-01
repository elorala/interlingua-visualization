import json
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, TapTool
from bokeh.plotting import figure, curdoc
from bokeh.transform import factor_cmap
from bokeh.layouts import layout

input_file = "data_umap_3lang.json"
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
p = figure(plot_width=800, plot_height=700, tools=tools, title='3 languages')

cr = p.circle('x', 'y',
              size=10, alpha=0.4,
              hover_color='yellow', hover_alpha=1.0,
              source=source_circle,
              color=factor_cmap('lang', ['red', 'blue', 'green'], ['en', 'fr', 'es']),
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

window = layout([[p, div]], sizing_mode='stretch_both')
curdoc().add_root(window)
