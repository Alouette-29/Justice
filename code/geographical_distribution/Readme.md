| 名称                        | 内容                                             | 备注                                        |
| --------------------------- | ------------------------------------------------ | ------------------------------------------- |
| 地理信息提取样本.txt        | 自动识别的样本的格式                             |                                             |
| unrecongnized_courtname.txt | 没有识别正确的法庭名称                           | 表现为 法院前面带有地区名                   |
| court_distribution.txt      | cpca自动抓取的所有法庭数据，相当于原数据集的切片 | txt文档，应当以表格读入,利用pd.read_csv（） |
|                             |                                                  |                                             |
|                             |                                                  |                                             |
|                             |                                                  |                                             |
|                             |                                                  |                                             |



这里做的效果还不太好,一方面是库没更新，一方面是有的法院不是按照行政区名称来的

可以先用结巴对地名进行提取，再拆分

可以利用的库有cpca , 

