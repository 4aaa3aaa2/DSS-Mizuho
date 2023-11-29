# DSS-Mizuho
2023年4月～7月期間、みずほリサーチ＆テクノロジーズでのデータサイエンス実践プロジェクトソースコードです。

みずほからのデータセキュリティの要求により、コードの部分しか公開出来ません。

# 環境
最初はLINUX環境のjupyter notebookで作成したものだが、ローカルPCのWindows（Python 3.10.9カーネル）にも実行できます。
コードで導入が必要なライブラリ以外、GPUで実行できるためのCUDAもインストール必要です。バーチャル環境の有無は構いません。

必要なライブラリ：  

[pandas](https://pandas.pydata.org/)  

[numpy](https://numpy.org/)  

[datetime](https://docs.python.org/ja/3/library/datetime.html)  

[matplotlib](https://matplotlib.org/)  

[seaborn](https://seaborn.pydata.org/)  

[Pytorch](https://pytorch.org/)  

[sklearn](https://scikit-learn.org/stable/)  

# データと各部分概要
データの概要：過去20年間記録した株価、債務、金利、為替レート、商品など100個以上の指標は変数で、みずほの自社の投資指標などは目標変数である。全てのデータは時系列データです。

１～４：データの概要と可視化、欠損値処理、日付関連情報追加、変数間の相関性可視化、変数重要性のランキング、統計的な特徴量抽出

５：LSTMモデル構築、特徴量により深層学習と予測、結果の評価

６：追加データの処理（主に特徴量作成）

７～９：LSTM精度向上のために、目標変数を適当に処理し、新しい目標変数に対する深層学習及びモデルの調整

予測の出力は目標変数の上昇・下落2分類と目標変数の数値予測ともあり。
