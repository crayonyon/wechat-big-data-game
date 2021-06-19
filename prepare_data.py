# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

END_DAY = 15

# 视频feed 本身的信息
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    train = pd.concat((train, temp), axis=1)
    return train

def prepare_data():
    feed_info_df = pd.read_csv(FEED_INFO)

    # user action df col : userid, date_, feedid,  "read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"
    #user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid", "device"] + FEA_COLUMN_LIST]

    feed_embed = pd.read_csv(FEED_EMBEDDINGS)

    test = pd.read_csv(TEST_FILE)

    # userid, date_, feedid, device, "read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    # ['userid', 'date_', 'feedid', device, 'read_comment', 'like', 'click_avatar', 'forward', 'comment', 'follow', 'favorite',
    # 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    # ['userid', 'feedid', 'device', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

    stage_day = 14
    # 基于userid统计的历史行为的次数
    user_date_feature_path = os.path.join(ROOT_PATH, "feature", "userid_feature.csv")
    user_date_feature = pd.read_csv(user_date_feature_path)
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature_path = os.path.join(ROOT_PATH, "feature", "feedid_feature.csv")
    feed_date_feature = pd.read_csv(feed_date_feature_path)
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    train = train.join(feed_date_feature, on=['feedid', 'date_'], how='left', rsuffix="_feed")
    train = train.join(user_date_feature, on=['userid', 'date_'], how='left', rsuffix='_user')
    # ['userid', 'date_', 'feedid', device, 'read_comment', 'like', 'click_avatar', 'forward', 'comment', 'follow', 'favorite',
    # 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', read_commentsum, likesum, click_avatarsum, forwardsum,
    #  read_commentsum_user, likesum_user, click_avatarsum_user, forwardsum_user]
    FEA_COL = ["read_comment", "like", "click_avatar", "forward"]
    feed_feature_col = [b + "sum" for b in FEA_COL]
    user_feature_col = [b + "sum_user" for b in FEA_COL]
    train[feed_feature_col] = train[feed_feature_col].fillna(0.0)
    train[user_feature_col] = train[user_feature_col].fillna(0.0)
    train[feed_feature_col] = np.log(train[feed_feature_col] + 1.0)
    train[user_feature_col] = np.log(train[user_feature_col] + 1.0)

    train[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 先全部加1，再填充未知用0
    train[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        train[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)

    user_sum_feature_path = os.path.join(ROOT_PATH, "feature", "userid_sum_feature.csv")
    user_sum_feature = pd.read_csv(user_sum_feature_path)
    user_sum_feature = user_sum_feature.set_index(['userid'])
    feed_sum_feature_path = os.path.join(ROOT_PATH, "feature", "feedid_sum_feature.csv")
    feed_sum_feature = pd.read_csv(feed_sum_feature_path)
    feed_sum_feature = feed_sum_feature.set_index(['feedid'])
    test = test.join(feed_sum_feature, on=['feedid'], how='left', rsuffix="_feed")
    test = test.join(user_sum_feature, on=['userid'], how='left', rsuffix='_user')
    test[feed_feature_col] = test[feed_feature_col].fillna(0.0)
    test[user_feature_col] = test[user_feature_col].fillna(0.0)
    test[feed_feature_col] = np.log(test[feed_feature_col] + 1.0)
    test[user_feature_col] = np.log(test[user_feature_col] + 1.0)

    test[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 先全部加1，再填充未知用0
    test[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        test[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)

    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    test.to_csv(ROOT_PATH + f'/test_data.csv', index=False)

    for action in tqdm(ACTION_LIST):
        # 以下两行先保留
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')

        # 负样本采样
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
        # 正负样本 合并
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        df_all.to_csv(ROOT_PATH + f'/train_data_for_{action}.csv', index=False)


def check_file():
    '''
    检查数据文件是否存在
    '''
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file


def create_dir():
    # 创建data目录
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s' % ROOT_PATH)
        os.mkdir(ROOT_PATH)
    # data目录下需要创建的子目录
    need_dirs = ["offline_train", "online_train", "evaluate", "submit",
                 "feature", "model", "model/online_train", "model/offline_train"]
    for need_dir in need_dirs:
        need_dir = os.path.join(ROOT_PATH, need_dir)
        if not os.path.exists(need_dir):
            print('Create dir: %s' % need_dir)
            os.mkdir(need_dir)


'''
为DeepFM 增加一些特征输入  用户过去各种行为次数sum  feed被交互行为次数 
'''
def statistic_feature(start_day=1, before_days=4, agg='sum'):
    """
    统计用户/feed 过去  before days = 5 的行为，进行sum，然后作为 DeepFM 的 dense 特征
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    :return:
    """
    # 统计初赛的所需要考虑的行为 过去n天的次数
    action_features = ["read_comment", "like", "click_avatar", "forward"]
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + action_features]
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + action_features]
        res_arr = []
        for start in range(start_day, END_DAY-before_days+1):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_days))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_days
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)

    # 统计初赛的所需要考虑的行为 过去14天的次数
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + action_features]
    for dim in ['userid', 'feedid']:
        print(dim)
        user_data = history_data[[dim, 'date_'] + action_features]
        temp = user_data.drop(columns=['date_'])
        temp['seen'] = 1
        temp = temp.groupby([dim]).agg(
            [agg]).reset_index()  # group by uid  uid 在 (start + before) day 过去 before days 各行为的次数 sum
        temp.columns = list(map(''.join, temp.columns.values))
        _path = os.path.join(ROOT_PATH, "feature", dim + "_sum_feature.csv")
        # print('Save to: $s'%_path)
        temp.to_csv(_path, index=False)

# def generate_sample():
#     day = 14
#     for action in ACTION_LIST:
#         # action_df =
#         pass

# 采用的是基本特征（离散特征：{'userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'}，连续特征：{'videoplayseconds'}）


def main():
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("请检查目录中是否存在下列文件: ", ",".join(not_exists_file))
        return
    statistic_feature()
    # sample_arr = generate_sample()
    prepare_data()


if __name__ == "__main__":
    main()
