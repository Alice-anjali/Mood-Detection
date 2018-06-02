from test_lyrics import magic_funct
from flask import Flask
from flask import render_template,request

#creates a Flask application, named app
app = Flask(__name__,static_url_path = '/static')

# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('index.html')
@app.route("/detect",methods = ['POST'])
def detect():
    lyrics = request.form['lyrics']
    mood = magic_funct(lyrics)
    return mood[0]
# run the application
if __name__ == "__main__":
    app.run(debug=True)
