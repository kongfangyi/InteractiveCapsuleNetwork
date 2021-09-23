import pandas as pd
import numpy as np
import os
import pickle
import heatMap_tool
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def nomalizationFun(leftAttValue, rightAttValue):
      y_tempStoreList = np.append(leftAttValue, rightAttValue)
      y_max = max(y_tempStoreList)
      y_min = min(y_tempStoreList)
      return ((leftAttValue - y_min)/(y_max - y_min)).tolist(), ((rightAttValue - y_min)/(y_max - y_min)).tolist()


'''
#中文可视化
label2emotion = {0: '惊讶', 1: '伤心', 2: '生气', 3: '开心', 4: '厌恶', 5: '害怕'}
emotion2label = {'惊讶': 0, '伤心': 1, '生气': 2, '开心': 3, '厌恶': 4, '害怕': 5}


contextContent = pd.read_table("D:\\codeBase\\chinese_periodical_2\\data\\context\\test.txt",sep='\t')
print(contextContent.columns)
print("载入数据大小：", contextContent.shape)

#singel情感分析
pickle_file = os.path.join('pickle', '1.5.sst_LSTM.pickle3')
pickle_file = open(pickle_file,'rb')
y_pred_att_single , y_pred_single = pickle.load(pickle_file)
y_pred_att_single = [i[i>0] for i in y_pred_att_single]

#context模型
pickle_file = os.path.join('pickle', '1.8.sst_contextCapsule_chinese.pickle3')
pickle_file = open(pickle_file,'rb')
y_pred_att_1, y_pred_att_2 , y_pred = pickle.load(pickle_file)
y_pred_att_1 = [i[i>0] for i in y_pred_att_1]
y_pred_att_2 = [i[i>0] for i in y_pred_att_2]
comLen = [(len(i[0]) + len(i[1])) for i in zip(y_pred_att_1,y_pred_att_2)]
'''
#英文可视化
label2emotion = {0: 'joy', 1: 'disgust', 2: 'surprise', 3: 'fear', 4: 'sad', 5: 'anger'}
emotion2label = {'joy': 0, 'disgust': 1, 'surprise': 2, 'fear': 3, 'sad': 4, 'anger': 5}

contextContent = pd.read_table("./data/wassa/test.csv",sep='\t')
contextContent.columns =['sentiment','reviews']
print(contextContent.columns)
print("载入数据大小：", contextContent.shape)

#single模型
pickle_file = os.path.join('pickle', 'sst_LSTM_wassa.pickle3')
pickle_file = open(pickle_file,'rb')
y_pred_att_single , y_pred_single = pickle.load(pickle_file)
y_pred_att_single = [i[i>0] for i in y_pred_att_single]

#context模型
pickle_file = os.path.join('pickle', '1.8.sst_contextCapsule_wassa.pickle3')
pickle_file = open(pickle_file,'rb')
y_pred_att_1, y_pred_att_2 , y_pred = pickle.load(pickle_file)
y_pred_att_1 = [i[i>0] for i in y_pred_att_1]
y_pred_att_2 = [i[i>0] for i in y_pred_att_2]
comLen = [(len(i[0]) + len(i[1])) for i in zip(y_pred_att_1,y_pred_att_2)]



#'''
#整体模型
x_tempStoreList = []
y_tempStoreList = []
countNum = 0
for i in y_pred_att_single:
      y_pred_att = i
      y_pred_att = y_pred_att.tolist()
      y_tempStoreList.append(y_pred_att)
      x_label = range(len(y_pred_att))
      x_tempStoreList.append(x_label)


      if(countNum!=0 and countNum%9 == 0):
            plt.figure()
            plt.title("single attention")
            for j in range(9):
                  plt.subplot(3, 3, j + 1)
                  plt.plot(x_tempStoreList[j], y_tempStoreList[j])
            plt.show()
            x_tempStoreList = []
            y_tempStoreList = []
      countNum += 1
      if countNum>18:
            break






#联合输出
x_tempStoreList_left = []
y_tempStoreList_left = []

x_tempStoreList_right = []
y_tempStoreList_right = []

y_tempStoreList = []
countNum = 0
for i in zip(y_pred_att_1,y_pred_att_2):
      y_pred_att_left = i[0]
      y_pred_att_right = i[1]

      #normalization
      y_tempStoreList = np.append(y_pred_att_left, y_pred_att_right)
      y_max = max(y_tempStoreList)
      y_min = min(y_tempStoreList)
      y_tempStoreList_left_normal = (y_pred_att_left - y_max)/(y_max - y_min)
      y_tempStoreList_right_normal = (y_pred_att_right - y_max)/(y_max - y_min)


      y_tempStoreList_left.append(y_tempStoreList_left_normal)
      y_tempStoreList_right.append(y_tempStoreList_right_normal)

      x_label_left = range(len(y_pred_att_left))
      x_tempStoreList_left.append(x_label_left)
      x_label_right = range(len(y_pred_att_right))
      x_tempStoreList_right.append(x_label_right)

      if(countNum!=0 and countNum%4 == 0):
            plt.figure()
            plt.title("context attention")
            for j in range(4):
                  plt.subplot(4, 2, 2*j + 1)
                  plt.plot(x_tempStoreList_left[j], y_tempStoreList_left[j])
                  plt.subplot(4, 2, 2 * j + 2)
                  plt.plot(x_tempStoreList_right[j], y_tempStoreList_right[j])
            plt.show()
            x_tempStoreList_left = []
            y_tempStoreList_left = []

            x_tempStoreList_right = []
            y_tempStoreList_right = []
      countNum += 1
      if countNum>18:
            break

#单独输出
x_tempStoreList = []
y_tempStoreList = []
countNum = 0
for i in zip(y_pred_att_1,y_pred_att_2):
      y_pred_att_left = i[0]
      y_pred_att_right = i[1]
      y_pred_att_left = y_pred_att_left.tolist()
      y_pred_att_right = y_pred_att_right.tolist()
      y_pred_att_left.extend(y_pred_att_right)
      y_tempStoreList.append(y_pred_att_left)
      x_label = range(len(y_pred_att_left))
      x_tempStoreList.append(x_label)

      if(countNum!=0 and countNum%9 == 0):
            plt.figure()
            plt.title("context attention")
            for j in range(9):
                  plt.subplot(3, 3, j + 1)
                  plt.plot(x_tempStoreList[j], y_tempStoreList[j])
            plt.show()
            x_tempStoreList = []
            y_tempStoreList = []
      countNum += 1
      if countNum>18:
            break
#'''
#载入内容数据
pickle_file = 'D:\\codeBase\\chinese_periodical_2\\pickle\\wassa_single_periodical.pickle3'

# revs为评论数据；W为gensim的权重矩阵;word_idx_map是单词Index索引；vocab是词典，每个单词出现的次数；maxlen是每句话中的单词数量
revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
cleaned_rev_cont = []
test_revs = [i['text'] for i in revs if(i['split']==-1)]
count_num = 1


while(count_num < 1000):
      inner_cleaned_rev = []
      rev_cont_list = test_revs[count_num].split()

      #获取关键词<#关键词#>的位置
      #keyword_pos = rev_cont_list.index("关键词")-2
      #获取去除关键词标识的内容
      #inner_cleaned_rev = rev_cont_list[:keyword_pos] + rev_cont_list[keyword_pos+5:]

      keyword_pos = rev_cont_list.index("TRIGGERWORD") - 2
      inner_cleaned_rev = rev_cont_list[:keyword_pos] + rev_cont_list[(keyword_pos + 4):]
      cleaned_rev_cont.append(inner_cleaned_rev)
      #获取去除关键词标识后attBigru模型的注意力值
      cleaned_pred_att_single = np.append(y_pred_att_single[count_num][:keyword_pos],y_pred_att_single[count_num][keyword_pos+5:])
      #将该注意力值进行均一化
      uniformed_clearned_pred_att_single = (cleaned_pred_att_single - min(cleaned_pred_att_single))/(max(cleaned_pred_att_single) - min(cleaned_pred_att_single))

      #上文内容及情感预测
      inner_cleaned_left_rev = rev_cont_list[:keyword_pos]

      #下文内容
      inner_cleaned_right_rev = rev_cont_list[keyword_pos+5:]
      #将上下文进行均一化
      uniformed_y_pred_att_left_loop, uniformed_y_pred_att_right_loop = nomalizationFun(y_pred_att_1[count_num],y_pred_att_2[count_num])
      uniformed_y_pred_att_left_loop.extend(uniformed_y_pred_att_right_loop)

      visual_data = [uniformed_clearned_pred_att_single,uniformed_y_pred_att_left_loop]
      ylabel = ["single attention", "context attention"]
      xlabel = inner_cleaned_rev
      heatMap_tool.draw_heatmap(visual_data, xlabel, ylabel)
      count_num += 1
      print("文字内容是",inner_cleaned_rev)




#---------------------------------------------------------------------------------------------------------
singleContent = ['不行', '了', '不行', '了', '，', '待会儿', '再', '收图', '我', '好', '<',
                 '#', '关键词', '#', '>', '呀', '～', '～', '～', '～', '为', '易燃', '装置',
                 '打爆', '电话', '全员', '晋级', '，', '你们', '最', '棒棒']
print("第51个：single",singleContent)
print("第51个：single 预测",label2emotion[y_pred_single[51]])
print("注意力值：",y_pred_att_single[51])
visual_data_single = np.append(y_pred_att_single[51][:10],y_pred_att_single[51][15:])
visual_data_single = (visual_data_single - min(visual_data_single))/(max(visual_data_single) - min(visual_data_single))

xlabel_sigle = singleContent[:10]+singleContent[15:]
#heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)

print("左边:"+contextContent["left"][51], "    右边："+contextContent["right"][51],
      "  情感:",contextContent['sentiment'][51])
con_left_51 = ['不行', '了', '不行', '了', '，', '待会儿', '再', '收图', '我', '好']
con_right_51 = ['呀', '～', '～', '～', '～', '为', '易燃', '装置', '打爆', '电话', '全员', '晋级', '，', '你们', '最', '棒棒']
print("左侧单词",con_left_51)
print("右侧单词",con_right_51)
y_pred_att_left = y_pred_att_1[51]
y_pred_att_right = y_pred_att_2[51]
print("左侧注意力值", y_pred_att_left)
print("右侧注意力值", y_pred_att_right)
print("预测情感：", label2emotion[y_pred[51]])

y_pred_att_left = y_pred_att_left.tolist()
y_pred_att_right = y_pred_att_right.tolist()

y_pred_att_left,y_pred_att_right = nomalizationFun(y_pred_att_left, y_pred_att_right)

y_pred_att_left.extend(y_pred_att_right)
y_pred_att_left = np.array(y_pred_att_left)
visual_data = [visual_data_single,y_pred_att_left]
xlabel = con_left_51 + con_right_51
ylabel = ["single attention","context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


#-----------------------------------------------------------------------------------------------------------
singleContent = ['这', '都', '哪些', '个', '好友', '搜', '的', '，', '我', 'tm', '想', '打死', '她', '看到', '其中', '几个', '字', '生理性', '<', '#', '关键词', '#', '>', '了']
print("第67个：single",singleContent)
print("第67个：single 预测",label2emotion[y_pred_single[67]])
print("注意力值：",y_pred_att_single[67])
visual_data_single = np.append(y_pred_att_single[67][:18],y_pred_att_single[67][23:])
visual_data_single = (visual_data_single - min(visual_data_single))/(max(visual_data_single) - min(visual_data_single))

xlabel_sigle = singleContent[:18]+singleContent[23:]
#heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)

print("左边:"+contextContent["left"][67], "    右边："+contextContent["right"][67],
      "  情感:",contextContent['sentiment'][67])
con_left_67 = ['这', '都', '哪些', '个', '好友', '搜', '的', '，', '我', 'tm', '想', '打死', '她', '看到', '其中', '几个', '字', '生理性']
con_right_67 = ['了']
print("左侧单词",con_left_67)
print("右侧单词",con_right_67)
y_pred_att_left = y_pred_att_1[67]
y_pred_att_right = y_pred_att_2[67]
print("左侧注意力值", y_pred_att_left)
print("右侧注意力值", y_pred_att_right)
print("预测情感：", label2emotion[y_pred[67]])

y_pred_att_left = y_pred_att_left.tolist()
y_pred_att_right = y_pred_att_right.tolist()

y_pred_att_left,y_pred_att_right = nomalizationFun(y_pred_att_left, y_pred_att_right)

y_pred_att_left.extend(y_pred_att_right)
y_pred_att_left = np.array(y_pred_att_left)
visual_data = [visual_data_single,y_pred_att_left]
xlabel = con_left_67 + con_right_67
ylabel = ["single attention","context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


#--------------------------------------------------------------------------------------------------------------------
singleContent = ['居然', '还有', '灯', '，', '太', '好看', '了', '，', '以后', '<', '#', '关键词', '#', '>', '了', '就', '打开', '看看', '，', '感觉', '能', '消气', '，', '但是', '你', '不能', '欺负', '我', '让', '我', '委屈', '太', '多次', '呢', '，', '因为', '只有', '520', '张小', '纸条', '噢']
print("第4个：single",singleContent)
print("第4个：single 预测",label2emotion[y_pred_single[4]])
print("注意力值：",y_pred_att_single[4])
visual_data_single = np.append(y_pred_att_single[4][:9],y_pred_att_single[4][14:23])
visual_data_single = (visual_data_single - min(visual_data_single))/(max(visual_data_single) - min(visual_data_single))

print("左边:"+contextContent["left"][4], "    右边："+contextContent["right"][4],
      "  情感:",contextContent['sentiment'][4])
con_left_4 = ['居然', '还有', '灯', '，', '太', '好看', '了', '，', '以后']
con_right_4 = ['了', '就', '打开', '看看', '，', '感觉', '能', '消气', '，', '但是', '你', '不能', '欺负', '我', '让', '我', '委屈', '太', '多次', '呢', '，', '因为', '只有', '520', '张小', '纸条', '噢']
print("左侧单词",con_left_4)
print("右侧单词",con_right_4[:9])
y_pred_att_left = y_pred_att_1[4]
y_pred_att_right = y_pred_att_2[4][:9]
print("左侧注意力值", y_pred_att_left)
print("右侧注意力值", y_pred_att_right)
print("预测情感：", label2emotion[y_pred[4]])

y_pred_att_left = y_pred_att_left.tolist()
y_pred_att_right = y_pred_att_right.tolist()
y_pred_att_left,y_pred_att_right = nomalizationFun(y_pred_att_left, y_pred_att_right)
y_pred_att_left.extend(y_pred_att_right)
y_pred_att_left = np.array(y_pred_att_left)
visual_data = [visual_data_single,y_pred_att_left]
xlabel = con_left_4 + con_right_4[:9]
ylabel = ["single attention","context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


#----------------------------------------------------------------------------------------------------------
singleContent = ['2018.6', '.', '122030', '虽然', '说', '你', '惹', '我', '<', '#', '关键词', '#', '>', '了', '但', '我', '也', '不', '应该', '把', '你', '锁', '在', '外面', '那么', '久', '好', '吧', '我', '承认', '是', '我', '的', '不', '对', '也', '就', '你', '能', '忍受', '我', '的', '臭', '脾气', '了']
print("第八个：single",singleContent)
print("第八个：single 预测",label2emotion[y_pred_single[8]])
print("注意力值：",y_pred_att_single[8][:-1])
visual_data_single = np.append(y_pred_att_single[8][:8],y_pred_att_single[8][13:-1])
visual_data_single = (visual_data_single - min(visual_data_single))/(max(visual_data_single) - min(visual_data_single))

print("第8个实例 左边:"+contextContent["left"][8], "    右边："+contextContent["right"][8],
      "  情感:",contextContent['sentiment'][8])
con_left_8 = ['2018.6', '.', '122030', '虽然', '说', '你', '惹', '我']
con_right_8 = ['了', '但', '我', '也', '不', '应该', '把', '你', '锁', '在', '外面', '那么', '久', '好', '吧', '我', '承认', '是', '我', '的', '不', '对', '也', '就', '你', '能', '忍受', '我', '的', '臭', '脾气']
print("左侧单词",con_left_8)
print("右侧单词",con_right_8)
print("左侧注意力值", y_pred_att_1[8])
print("右侧注意力值", y_pred_att_2[8][:-1])
print("预测情感：", label2emotion[y_pred[8]])

y_pred_att_left = y_pred_att_1[8].tolist()
y_pred_att_right = y_pred_att_2[8][:-1].tolist()

y_pred_att_left,y_pred_att_right = nomalizationFun(y_pred_att_left, y_pred_att_right)
y_pred_att_left.extend(y_pred_att_right)
visual_data = [visual_data_single , y_pred_att_left]
xlabel = con_left_8 + con_right_8
ylabel = ["single attention","context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


#-----------------------------------------------------------------------------------------------------------
singleContent = ['我', '从未', '如', '此刻', '这般', '，', '热爱', '又', '<', '#', '关键词', '#', '>', '生活', '。', '生活', '的', '意义', '是', '什么', '「', '和', '每天', '讲', '再见', '」', '生活', '的', '意义', '是', '什么', '「', '和', '每天', '讲', '再见', '」', '(', '来自', ')']
print("第九个：single",singleContent)
print("第九个：single 预测",label2emotion[y_pred_single[9]])
print("注意力值：",y_pred_att_single[9])
visual_data = [y_pred_att_single[9]]
xlabel = singleContent
ylabel = ["single attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


print("第9个实例 左边:"+contextContent["left"][9], "    右边："+contextContent["right"][9],
      "  情感:",contextContent['sentiment'][9])
con_left_9 = ['我', '从未', '如', '此刻', '这般', '，', '热爱', '又']
con_right_9 =['生活', '。', '生活', '的', '意义', '是', '什么', '「', '和', '每天', '讲', '再见', '」', '生活', '的', '意义', '是', '什么', '「', '和', '每天', '讲', '再见', '」', '(', '来自', ')']
print("左侧单词",con_left_9)
print("右侧单词",con_right_9)
print("左侧注意力值", y_pred_att_1[9])
print("右侧注意力值", y_pred_att_2[9])
print("预测情感：", label2emotion[y_pred[9]])

y_pred_att_left = y_pred_att_1[9].tolist()
y_pred_att_right = y_pred_att_2[9].tolist()
y_pred_att_left.extend(y_pred_att_right)
visual_data = [y_pred_att_left]
xlabel = con_left_9 + con_right_9
ylabel = ["context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


#----------------------------------------------------------------------------------------------------------------
singleContent = ['努力', '的', '意义', '在于', '你', '随时', '可以', '逃出', '<', '#', '关键词', '#', '>', '的', '圈子']
print("第922个：single",singleContent)
print("第922个：single 预测",label2emotion[y_pred_single[923]])
print("注意力值：",y_pred_att_single[923])
visual_data = [y_pred_att_single[923]]
xlabel = singleContent
ylabel = ["single attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)


print("左边:"+contextContent["left"][922], "    右边："+contextContent["right"][922],
      "  情感:",contextContent['sentiment'][922])
con_left_922 = ['努力', '的', '意义', '在于', '你', '随时', '可以', '逃出']
con_right_922 = ['的', '圈子']
print("左侧单词",con_left_922)
print("右侧单词",con_right_922)
y_pred_att_left = y_pred_att_1[922]
y_pred_att_right = y_pred_att_2[922]
print("左侧注意力值", y_pred_att_left)
print("右侧注意力值", y_pred_att_right)
print("预测情感：", label2emotion[y_pred[922]])

y_pred_att_left = y_pred_att_left.tolist()
y_pred_att_right = y_pred_att_right.tolist()
y_pred_att_left.extend(y_pred_att_right)
visual_data = [y_pred_att_left]
xlabel = con_left_922 + con_right_922
ylabel = ["context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)



##预测错误：生气--》伤心
print("左边:"+contextContent["left"][329], "    右边："+contextContent["right"][329],
      "  情感:",contextContent['sentiment'][329])
con_left_329 = ['千万', '不要']
con_right_329 = ['死', '得', '快', '！', '！', '！', '！', '！', '！', '！', '！']
print("左侧单词",con_left_329)
print("右侧单词",con_right_329)
print("左侧注意力值", y_pred_att_1[329])
print("右侧注意力值", y_pred_att_2[329])
print("预测情感：", label2emotion[y_pred[329]])

#预测错误：伤心--》开心
print("左边:"+contextContent["left"][742], "    右边："+contextContent["right"][742],
      "  情感:",contextContent['sentiment'][742])
con_left_742 = ['看着', '情人', '一对对', '，', '我', '他', '妈', '真的', '好']
con_right_742 = ['。', '别拦', '着', '我']
print("左侧单词",con_left_742)
print("右侧单词",con_right_742)
print("左侧注意力值", y_pred_att_1[742])
print("右侧注意力值", y_pred_att_2[742])
print("预测情感：", label2emotion[y_pred[742]])


#预测错误：伤心--》厌恶
print("左边:"+contextContent["left"][7], "    右边："+contextContent["right"][7],
      "  情感:",contextContent['sentiment'][7])
con_left_7 = ['感觉', '无力', '，', '感觉', '到', '绝望', '，']
con_right_7 = ['，', '听力', '虐死', '我', '啦', '，', '我', '应该', '怎么办', '腻', '？', '没关系', '，', '肉肉', '永远', '是', '小可爱', '，', '明天', '早起', '背', '听力', '托福', '一定', '90', '+++++']
print("左侧单词",con_left_7)
print("右侧单词",con_right_7)
print("左侧注意力值", y_pred_att_1[7])
print("右侧注意力值", y_pred_att_2[7])
print("预测情感：", label2emotion[y_pred[7]])

y_pred_att_left = y_pred_att_1[7].tolist()
y_pred_att_right = y_pred_att_2[7].tolist()
y_pred_att_left.extend(y_pred_att_right)
visual_data = [y_pred_att_left]
xlabel = con_left_7 + con_right_7
ylabel = ["context attention"]
heatMap_tool.draw_heatmap(visual_data,xlabel,ylabel)

