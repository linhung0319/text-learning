# Text Learning

>Enron Dataset 包含許多email文件，從中選取寄信者來自Crist和Sara的郵件，對一部份來自Crist和Sara的郵件作Machine Learning，試圖建構一個模型，在不知道寄件者的情況下，僅憑郵件的內容，便可判別此郵件來自Crist或Sara
>
>更詳細對此程式演算法的解釋請點[這裡](https://linhung0319.github.io/text-learning/)

## Getting Started

>本程式以python3執行，並使用函式庫nltk, numpy, scipy, sklearn

**以下有兩種方式執行程式:**

- 下載[Enron Dataset](https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz)：
  - 執行 **startup.py** ，下載並解壓縮Enron Dataset至./maildir
  - 執行 **pickle_email_text.py** ，將原始資料處理為 **word_data.pkl** 和 **email_authors.pkl**
  - 使用 **word_data.pkl** 和 **email_authors.pkl** ，再執行 **svm** 資料夾內的主程式 **svm_author_id.py**

- 直接使用已處理過後的資料：
  - 使用 **word_data.pkl** 和 **email_authors.pkl** ，並直接執行 **svm** 資料夾內的主程式 **svm_author_id.py**

## Preprocess

>將Enron Dataset 中來自Crist和Sara的郵件進行處理
>
>1. 只留下郵件中的內容部份
>
>2. 將內容部份的文字進行Stem和TFIDF轉換，得到每個字詞的重要性權重

- 執行 **pickle_email_text.py** 
  - 將email內容的文字進行Stem轉換 ( **parse_out_email_text.py** )
  - 將轉換後的文字以list的形式儲存成 **word_data.pkl**
  - 每個字對應的作者以list的形式儲存成 **email_authors.pkl**
  
- **email_preprocess.py**   
  - 讀取 **word_data.pkl** 和 **email_authors.pkl** 並將這些字詞進行TFIDF轉換，計算出每個字詞的重要性權重
  - 刪除掉其中出現過於頻繁的字詞(過於頻繁的字詞不具備代表性)
  - 將字詞與其作者分為Training Set 和 Testing Set

## SVM

>利用Support Vector Machine演算法建構模型，根據Email使用的文字判斷Email屬於Sara或是Chris
>
>1. 調整SVM的參數以找到分類精準度最佳的結果 (Tuning Parameters in SVM)
>
>2. 利用SVM的Hyper Plane係數，找出對分類具有最高判別力的文字詞 （Top Features）
>
>3. 計算SVM分類結果的精準度 （Accuracy Score, Confusion Matrix, Precision, Recall Score）

- 執行 **svm_author_id.py**
  - 測試SVM的不同參數，以得到最高的分類精準度 （**find_best_param**）
  - 以最佳的SVM參數去建構分類Email的模型
  - 找出此SVM模型，作為判斷Email分類，最重要的文字詞
  - 計算模型的精準度
  
- **plot_gallery.py**
  - 畫出不同參數下SVM分類Email的精準度
  - 分別畫出SVM模型判斷Email屬於Sara或Chris，最重要的文字詞及其相對比重
