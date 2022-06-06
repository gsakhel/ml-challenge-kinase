from flask import Flask, render_template, Markup

app  = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template("index.html")

    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)