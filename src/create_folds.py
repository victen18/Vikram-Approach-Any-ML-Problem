import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\vikra\PycharmProjects\vikram_ml\Project\input\train.csv')

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f

    df.to_csv(r'C:\Users\vikra\PycharmProjects\vikram_ml\Project\input\mnist_train_folds.csv',index=False)
