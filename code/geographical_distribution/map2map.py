#the name of this file is humuorously using two meaning of a word
#the first map is verb
#the second map is noun
#this script can decode a string into address so that we can locate whrere the case takes place

# it is a simple version 

#package one 
from chinese_province_city_area_mapper.transformer import CPCATransformer
Ca = CPCATransformer()
Ca.transform(['贵州省黔南布依族苗族自治州中级人民法院',
              '湖南省岳阳市岳阳楼区人民法院',
              '湖南省隆回县人民法院',
              '贵州省都匀市人民法院',
              '孙吴县人民法院',
              '原阳县人民法院'])
################################################################
#  省	市	区                                                  #                         
# 0			                                                   #
# 1	湖南省	岳阳市	岳阳楼区                                     #
# 2	湖南省	邵阳市	隆回县                                       #
# 3	贵州省		都匀市                                          #
# 4	黑龙江省	黑河市	孙吴县                                   #
# 5	河南省	新乡市	原阳县                                       #
#################################################################
#pakage two 
import cpca
area_str = ['贵州省黔南布依族苗族自治州中级人民法院',
              '湖南省岳阳市岳阳楼区人民法院',
              '湖南省隆回县人民法院',
              '贵州省都匀市人民法院',
              '孙吴县人民法院',
              '原阳县人民法院']
df_str = cpca.transform(area_str )
df_str

################################################################
# 	省	市	区	地址	adcode                                  #
# 0	贵州省	黔南布依族苗族自治州	None	中级人民法院522700    #
# 1	湖南省	岳阳市	岳阳楼区	人民法院	430602               #
# 2	湖南省	邵阳市	隆回县	人民法院	430524                   #
# 3	贵州省	黔南布依族苗族自治州	都匀市	人民法院	522701    #
# 4	黑龙江省	黑河市	孙吴县	人民法院	231124               #
# 5	河南省	新乡市	原阳县	人民法院	410725                   #
################################################################