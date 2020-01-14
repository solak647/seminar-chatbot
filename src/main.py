import numpy as np
import pandas as pd
def main():
    df = pd.read_csv('./data/Essays/essays2007.csv')
    print("The size of data is {0}".format(df.shape[0]))
    data = df['text'].astype(str).values.tolist()
    labels = df['cArgs'].values

if __name__ == "__main__":
    main()