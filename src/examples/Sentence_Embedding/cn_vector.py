if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import numpy as np
    from tensorflow.keras.layers import *
    from layers.bert import load_pretrained_model
    from layers.utils import load_vocab, SimpleTokenizer

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    config_path = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_config.json"
    checkpoint_path = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_model.ckpt"
    vocab_file = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/vocab.txt"

    max_seq_length = 128

    maxlen = 100
    config_path = '/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/vocab.txt'

    token_dict = load_vocab(dict_path)  # 读取词典
    tokenizer = SimpleTokenizer(token_dict)  # 建立分词器
    bert_model = load_pretrained_model(config_path, checkpoint_path)  # 建立模型，加载权重

    # 编码测试
    token_ids, segment_ids = tokenizer.encode(u'语言模型')
    print(bert_model.predict([np.array([token_ids]), np.array([segment_ids])]).shape)

    # save_h5_model(model)
    # tf.keras.experimental.export_saved_model(model, saved_model_path)

"""

train examles:
0	选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般	None	1
1	15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错	None	1
2	房间太小。其他的都一般。。。。。。。。。	None	0
3	1.接电源没有几分钟,电源适配器热的不行. 2.摄像头用不起来. 3.机盖的钢琴漆，手不能摸，一摸一个印. 4.硬盘分区不好办.	None	0
4	今天才知道这书还有第6卷,真有点郁闷:为什么同一套书有两种版本呢?当当网是不是该跟出版社商量商量,单独出个第6卷,让我们的孩子不会有所遗憾。	None	1


dev examles:
0	這間酒店環境和服務態度亦算不錯,但房間空間太小~~不宣容納太大件行李~~且房間格調還可以~~ 中餐廳的廣東點心不太好吃~~要改善之~~~~但算價錢平宜~~可接受~~ 西餐廳格調都很好~~但吃的味道一般且令人等得太耐了~~要改善之~~	None	1
1	<荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!	None	1
2	商品的不足暂时还没发现，京东的订单处理速度实在.......周二就打包完成，周五才发货...	None	0
3	２００１年来福州就住在这里，这次感觉房间就了点，温泉水还是有的．总的来说很满意．早餐简单了些．	None	1
4	不错的上网本，外形很漂亮，操作系统应该是个很大的 卖点，电池还可以。整体上讲，作为一个上网本的定位，还是不错的。	None	1


test examles:
0	这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般	None	1
1	怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是不是赠送的个别现象，可是后来发现每张DVD后面都有！真不知道生产商怎么想的，我想看的是猫和老鼠，不是米老鼠！如果厂家是想赠送的话，那就全套米老鼠和唐老鸭都赠送，只在每张DVD后面添加一集算什么？？简直是画蛇添足！！	None	0
2	还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。	None	0
3	交通方便；环境很好；服务态度很好 房间较小	None	1
4	不错，作者的观点很颠覆目前中国父母的教育方式，其实古人们对于教育已经有了很系统的体系了，可是现在的父母以及祖父母们更多的娇惯纵容孩子，放眼看去自私的孩子是大多数，父母觉得自己的孩子在外面只要不吃亏就是好事，完全把古人几千年总结的教育古训抛在的九霄云外。所以推荐准妈妈们可以在等待宝宝降临的时候，好好学习一下，怎么把孩子教育成一个有爱心、有责任心、宽容、大度的人。	None	1



Converting examples to features: 100%|████| 9600/9600 [00:07<00:00, 1236.69it/s]
Converting examples to features: 100%|████| 1200/1200 [00:00<00:00, 1366.45it/s]

______________________________________________________________________________________________________________________________________________________
Layer (type)                                     Output Shape                     Param #           Connected to                                      
======================================================================================================================================================
input_ids (InputLayer)                           (None, 128)                      0                                                                   
______________________________________________________________________________________________________________________________________________________
segment_ids (InputLayer)                         (None, 128)                      0                                                                   
______________________________________________________________________________________________________________________________________________________
model (Model)                                    multiple                         101677056         input_ids[0][0]                                   
                                                                                                    segment_ids[0][0]                                 
______________________________________________________________________________________________________________________________________________________
lambda (Lambda)                                  (None, 768)                      0                 model[1][0]                                       
______________________________________________________________________________________________________________________________________________________
dense_72 (Dense)                                 (None, 1)                        769               lambda[0][0]                                      
======================================================================================================================================================
Total params: 101,677,825
Trainable params: 101,677,825
Non-trainable params: 0
______________________________________________________________________________________________________________________________________________________
Train on 9600 samples, validate on 1200 samples

BERT 参数训练，Adam(lr = 1e-5), GPU::
Epoch 1/10
9600/9600 [==============================] - 425s 44ms/sample - loss: 0.2477 - acc: 0.9067 - val_loss: 0.1960 - val_acc: 0.9267
Epoch 2/10
9600/9600 [==============================] - 418s 44ms/sample - loss: 0.1204 - acc: 0.9598 - val_loss: 0.1908 - val_acc: 0.9383
Epoch 3/10
9600/9600 [==============================] - 419s 44ms/sample - loss: 0.0626 - acc: 0.9780 - val_loss: 0.2130 - val_acc: 0.9425
Epoch 4/10
9600/9600 [==============================] - 420s 44ms/sample - loss: 0.0362 - acc: 0.9865 - val_loss: 0.2577 - val_acc: 0.9408

BERT 参数固定：Adam(lr = 0.001), GPU: 741M:

"""
