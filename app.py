from flask import Flask, render_template,request
import modeling

app = Flask(__name__)


@app.route('/', methods=('GET','POST'))
def main():  # put application's code here
    data = None
    if request.method == 'POST':
        text = request.form["text"]
        if text:
            data = modeling.summarize_text(text)
    return render_template("index.html", data=data)

if __name__ == '__main__':
    app.run()
