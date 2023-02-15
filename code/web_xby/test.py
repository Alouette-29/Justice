from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def test():
    user = {"username":"Alouette", "email":"123456@163.com"}
    return render_template("index.html", user=user)

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8000)
    app.run(debug=True)