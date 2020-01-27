import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv('../data/myPersonality/mypersonality_final.csv', encoding='latin1')
    print("The size of data is {0}".format(df.shape[0]))
    labels = ['cEXT','cNEU','cAGR','cCON','cOPN']
    for label in labels:
        df[label] = df[label] == 'y'
        df[label] = df[label].astype(int)
    df['labels'] = df[labels].values.tolist()
    df2 = pd.read_csv('../data/Essays/essays2007.csv', encoding='latin1')
    print("The size of data is {0}".format(df2.shape[0]))
    df2['labels'] = df2[labels].values.tolist()
    df4 = df[['STATUS','labels']]
    df4 = df4.rename(columns={'STATUS': "text"})
    df3 = df2[['text','labels']]
    data = df4.append(df3, ignore_index=True)
    train_size = 0.8
    train_dataset=data.sample(frac=train_size,random_state=200).reset_index(drop=True)
    test_dataset=data.drop(train_dataset.index).reset_index(drop=True)

    train_df = pd.DataFrame(train_dataset, columns=['text', 'labels'])
    eval_df = pd.DataFrame(test_dataset)
    return train_df, eval_df

def get_data_onelabel(label):
    df = pd.read_csv('../data/myPersonality/mypersonality_final.csv', encoding='latin1')
    print("The size of data is {0}".format(df.shape[0]))
    df[label] = df[label] == 'y'
    df[label] = df[label].astype(int)
    df2 = pd.read_csv('../data/Essays/essays2007.csv', encoding='latin1')
    print("The size of data is {0}".format(df2.shape[0]))
    df4 = df[['STATUS',label]]
    df4 = df4.rename(columns={'STATUS': "text", label: 'labels'})
    df2 = df2.rename(columns={label: 'labels'})
    df3 = df2[['text','labels']]
    data = df4.append(df3, ignore_index=True)
    train_size = 0.8
    #train_dataset=data.sample(frac=train_size).reset_index(drop=True)
    #test_dataset=data.drop(train_dataset.index).reset_index(drop=True)
    x_train, x_test, y_train, y_test = train_test_split(data['text'],data['labels'], test_size=0.2, random_state=200)
    d = {'text': x_train, 'labels': y_train}
    dd = {'text': x_test, 'labels': y_test}
    train_df = pd.DataFrame(d, columns=['text', 'labels'])
    eval_df = pd.DataFrame(dd, columns=['text', 'labels'])
    return train_df, eval_df