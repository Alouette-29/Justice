from flask import Flask, render_template, request
from analyze import analysize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    content = request.form['text']
    classify, similar, summary = analysize(content)
    return render_template('result.html', classify=classify, similar=similar, summary=summary)


if __name__ == '__main__':
    app.run(debug=True)
