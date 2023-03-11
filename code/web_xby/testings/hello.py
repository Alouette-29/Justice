from flask import Flask, request, render_template
from datetime import datetime

'''
使用Flask类创建一个对象, 叫做app
__name__: 代表hello.py这个模块
此参数可以帮助快速定位bug, 对于寻找模板文件有一个相对路径
'''
app = Flask(__name__)

'''
创建一个路由和视图函数的映射
'/'叫做跟路由
'''
@app.route("/")
def hello_world():
    return "<p>Hello, world!</p>"

@app.route("/profile")
def profile():
    return "<p>This is the profile page!</p>"

@app.route("/blog/<int:blog_id>")  # 可以规定参数类型
def blog_detail(blog_id):
    return "The blog you are reading is from author %s" % blog_id

# 如果路径形如/book/list?page=3, 表示要看第三页
@app.route("/book/list")
def book_list():
    # request.args 类字典类型
    page = request.args.get(key="page", default=1, type=int)
    return f"This is book list page number{page}"


# 关于过滤器的使用：以时间显示为例
def datetime_format(value, format="%Y年%m月%d日 %H:%M"):
    return value.strftime(format)

app.add_template_filter(datetime_format, "dformat")

@app.route("/filter")
def filter_demo():
    user = {"username":"Alouette", "email":"123456@163.com"}
    time = datetime.now()
    return render_template("filter.html", user=user, time=time)


# 关于Jinja2中的控制语句
@app.route("/control")
def control_statement():
    age = 17
    books = [{"name":"红楼梦", "author":"曹雪芹"}, {"name":"水浒传", "author":"施耐庵"}]
    return render_template("control.html", age=age, books=books)


# Jinja2的模板继承
@app.route("/child1")
def child1():
    return render_template("child1.html")


# Jinja2加载静态文件
@app.route("/static")
def static_demo():
    return render_template("static.html")


# 1. debug模式: 修改代码实时在网页看到变化, 无需关了重跑
# 2. 修改host: 其他电脑也能访问我的flask项目, 只要连同一个网
# 3. 修改port端口号(默认5000)

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8000)
    app.run(debug=True)
