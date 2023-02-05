

## 文本预处理

> 洗数据  <br>
>
> 先空值列清除 bk<br>
>
> is_private 删除 ——bk<br>
>
> crawl time、d 删除 --bk<br>
>
> // dict 里面 怎么样批量选择 Keys （遍历选择） 
>
> // 新建空列表 二维列表  里面的一维 append(key-value)
>
> 
>
> 
>
> 计算 content - texts 列同时为空的情况 --why 
>
>  'basics_text': 501696,
>
>  'judge_record_text': 877445,
>
>  'head_text': 889256,
>
>  'tail_text': 888968,
>
>  'judge_result_text': 844359,
>
>  'judge_reson_text': 843756,
>
>  'content': 889502,
>
> u3000清理一下， \n\n\n\n清理
>
> 1125799->23W
>
> 
>
> --xby 
>
> 看一下每一列有多少个取值
>
> court_id head                                      已完成
>
> appellor 去重                                       已完成
>
> keywords 去none                               0条有效，需要自己提取关键词来补全吗？不过已经有了`self.dictionary`，可以在里面设置
>
> case_type                                             1125799条有效
>
> cause                                                    1125799条有效
>
> trial_round                                           1125349条有效
>
> settle_type                                           0条有效，似乎也需要在原文提取
>