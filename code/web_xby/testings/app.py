from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)


# MySQL所在的主机名
HOSTNAME = "127.0.0.1"
# MySQL监听的端口号, 默认3306
PORT = 3306
# 连接MySQL的用户名密码, 用户可以自己设置
USERNAME = "root"
PASSWORD = "4-gmfgUcUut9aZq"
# MySQL上创建的数据库名称
DATABASE = "database_learn"

app.config['SQLALCHEMY_DATABASE_URI'] = \
    f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"


# 在app.config中设置好连接数据库的信息，然后使用SQLAlchemy(app)创建一个db对象
# SQLAlchemy会自动读取app.config中设置的信息
db = SQLAlchemy(app)

# 建立与数据库的映射
migrate = Migrate(app, db)

'''
ORM模型映射成表的三步
flask db init   只需要执行一次
flask db migrate    识别ORM模型的改变, 生成迁移脚本
flask db upgrade    运行迁移脚本, 同步到数据库中
'''


# 测试数据库是否连接成功
# with app.app_context():
#     with db.engine.connect() as conn:
#         rs = conn.execute("select 1")
#         print(rs.fetchone())
#         # 如果输出(1,)说明连接成功


class User(db.Model):
    __tablename__ = "user"
    # id是主键无需手动设置而是自动加一
    id = db.Column(db.Integer, primary_key=True, autoincrement=True) 
    # 被映射为varchar类型, 需要指定最大长度(100), 字段不能为空
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    #---------------------------------------------------------------
    # articles是Article类的对象, 可以用author访问到
    # articles = db.relationship("Article", back_populates="author")
    #---------------------------------------------------------------


class Article(db.Model):
    __tablename__ = "article"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    # 添加作者的外键(Integer类型是因为要和id对应)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))  # user这张表中的id字段
    #---------------------------------------------------------------------------------
    # author是User类的对象, 可以直接用author.articles访问文章
    # author = db.relationship("User", back_populates="articles")  # 反向引用, 和前面对应
    #---------------------------------------------------------------------------------
    author = db.relationship("User", backref="articles")  # 会自动给User类绑定一个articles属性


# 建立与数据库的映射, 但是检测不到字段增删, 只是初学暂用的方法, 后面替换为migrate
# with app.app_context():
#     db.create_all()


@app.route("/")
def home():
    # user = {"username":"Alouette", "email":"123456@163.com"}
    # return render_template("index.html", user=user)
    return render_template("static.html")


@app.route("/user/add")
def add_user():
    # 1. 创建ORM对象
    user = User(username="Alouette", password="111111")
    # 2. 把ORM对象添加到db.session中
    db.session.add(user)
    # 3. 把db.session中的改变同步到数据库中
    db.session.commit()
    return "用户创建成功"


@app.route("/user/query")
def query_user():
    # 1. get, 根据主键查找
    # user = User.query.get(1)
    # print(f"{user.id}:{user.username}-{user.password}")
    # return "用户查找成功"

    # 2. filter_by查找, 得到一个Query对象
    users = User.query.filter_by(username="Alouette")
    for user in users:
        print(user.username)
    return "用户查找成功"


@app.route("/user/update")
def update_user():
    user = User.query.filter_by(username="Alouette").first()
    user.password = "222222"
    db.session.commit()
    return "用户更新成功"


@app.route("/user/delete")
def delete_user():
    user = User.query.filter_by(username="Alouette").first()
    db.session.delete(user)
    db.session.commit()
    return "用户删除成功"


@app.route("/article/add")
def add_article():
    article1 = Article(title="flask学习大纲", content="balabala")
    article1.author = User.query.filter_by(username="Alouette").first()
    article2 = Article(title="Django学习大纲", content="balabalabala")
    article2.author = User.query.filter_by(username="Alouette").first()

    db.session.add_all([article1, article2])
    db.session.commit()
    return "文章添加成功"


@app.route("/article/query")
def query_article():
    user = User.query.filter_by(username="Alouette").first()
    for article in user.articles:
        print(article.title)
    return "文章查找成功"


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8000)
    app.run()