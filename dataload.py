import os
import pandas as pd
import matplotlib.pyplot as plt

#Data preprocessing
file_list = os.listdir('./input/')
target_cell_list = [4458,4459,4460,4558,4559,4560,4658,4659]
df_all = pd.DataFrame({})

if os.path.exists("dataset.csv"):
    df_all = pd.read_csv("dataset.csv")
    df_all = df_all.set_index('minute').sort_index()
    print(df_all.head())
else:
    for i,f in enumerate(sorted(file_list)):
        if 'sms-call-internet-mi-' in f:
            df = pd.read_csv('./input/' + f, parse_dates=['datetime'])
            df = df.fillna(0)
            df['minute'] = df.datetime.dt.minute + df.datetime.dt.hour*60 + (df.datetime.dt.day)*24*60 + (df.datetime.dt.month - 11) * 60*24*30
            df = df[['minute','CellID','internet', 'smsin','smsout', 'callin','callout']].groupby(['minute', 'CellID'], as_index=False).sum()
            df_all = pd.concat([df_all, df])
            print(f)
    df_all = df_all.set_index('minute').sort_index()
    print(df_all.head())
    df_all.to_csv('dataset.csv')

if not os.path.exists("up_ratio.csv"):
    df_in = pd.DataFrame({})
    df_out = pd.DataFrame({})
    df_in_all = pd.DataFrame({})
    df_out_all = pd.DataFrame({})

    for cell in target_cell_list:
        df = df_all[df_all.CellID == cell]['smsin']
        if len(df_in_all) == 0:
            df_in_all['sms'] = df
        else:
            df_in_all = pd.merge(df_in_all, df, left_index=True,
                                 right_index=True, suffixes=("", str(cell)))


        if len(df_in) == 0:
            df_in = df
        else:
            df_in = df_in + df
    df_in_all.columns = ["0", "1", "2", "3", "4", "5", "6", "7"]

    for cell in target_cell_list:
        df = df_all[df_all.CellID == cell]['smsout']
        if len(df_out_all) == 0:
            df_out_all['sms'] = df
        else:
            df_out_all = pd.merge(df_out_all, df, left_index=True,
                                 right_index=True, suffixes=("", str(cell)))

        if len(df_out) == 0:
            df_out = df
        else:
            df_out = df_out + df

    df_out_all.columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    df_out_ratio = (df_out * 100 / (df_in + df_out))
    df_out_ratio.to_csv('up_ratio.csv')
    df_in_all.to_pickle('down_sms.pkl')
    df_out_all.to_pickle('up_sms.pkl')

    f = plt.figure()
    df_in.plot()
    df_out.plot()
    plt.show()
else:
    df_out_ratio = pd.read_csv("up_ratio.csv")

