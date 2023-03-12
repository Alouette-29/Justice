from flask import Flask, render_template, request
from analysize import analysize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    content = request.form['text']
    report = analysize(content)
    # print(type(report))  # str
    return render_template('result.html', report=report)

if __name__ == '__main__':
    app.run(debug=True)
