from flask import Flask, render_template, request
from analysize import analysize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result1', methods=['POST'])
def result1():
    content = request.form['text']
    report = analysize(content)
    # print(type(report))  # str
    return render_template('result.html', report=report)
	

@app.route('/result2', methods=['POST'])
def result2():
    # 用户通过文件提交
    content = request.data.decode('utf-8')
    report = analysize(content)
    # print(type(report))  # str
    return render_template('result.html', report=report)


if __name__ == '__main__':
    app.run()
