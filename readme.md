# Interlingua Visualization

**Requirements :**
* Python 3.7.2
* Bokeh >= 1.0.4
* Flask 1.0.3

In order to run the website, you need to run the bokeh server first. 

## Step 1
To run the app with the intermediate representations, first execute this command at the root of the repository : 
```
 bokeh serve --port 5200 --allow-websocket-origin localhost:8080 --allow-websocket-origin 127.0.0.1:8080 app/myapp.py
```

## Step 2
To run the app with the decoders layers, run this second process at the root of the repository :  
```
 bokeh serve --port 5201 --allow-websocket-origin localhost:8080 --allow-websocket-origin 127.0.0.1:8080 app/decoder_app.py
```

## Step 3 
To run the main page, you need to execute this last command at the same place : 
```
python app/main_page.py
```

Now, you can navigate by using your browser at http://127.0.0.1:8080/

