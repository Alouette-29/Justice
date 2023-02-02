Source code puts here.

2023-1-28



# 文件列表

| 文件名        | 命令                 | 作用                                                         |
| ------------- | -------------------- | ------------------------------------------------------------ |
| reorganize.py | python reorganize.py | 将大文件拆成一个文件夹，可以方便处理// 然而我发现并不方便处理，还是算了吧 |
| deduction.py  | please read parse()  | 一个别人的中文的命名实体识别模型,或许可以用来识别 人名地名组织名 |
| map2map.py    | -                    | 识别地名，不过比较传统的算法                                 |
| mask_entity   | -                    | 识别人名，并且替换成中立的名称，目前使用甲乙，也可使用张三李四 |
|               |                      |                                                              |

## reorganize.py

requirements: no extra packages need to be installed

1. create a new folder to place reorganze.py and p2-1-2021.txt
2. open cmd under this folder

```
! python reorganze.py
```