# Text Learning

>Enron Dataset 包含許多email文件，從中選取寄信者來自Crist和Sara的郵件，對一部份來自Crist和Sara的郵件作Machine Learning，試圖建構一個模型，在不知道寄件者的情況下，僅憑郵件的內容，便可判別此郵件來自Crist或Sara
>
>更詳細對此程式演算法的解釋請點[這裡](https://linhung0319.github.io/text-learning/)

## Getting Started

## Preprocess

>將Enron Dataset 中來自Crist和Sara的郵件進行處理
>
>1. 只留下郵件中的內容部份
>
>2. 將內容部份的文字進行Stem和TFIDF轉換，得到每個字詞的重要性權重

- 執行 **vectorize_text.py** 

  - 得到所有email內容的文字儲存而成的list **word_data.pkl**，和每個字對應的作者 **email_authors.pkl**

**email_preprocess.py** 會讀取 **word_data.pkl** 和 **email_authors.pkl** 並將這些字詞進行TFIDF轉換，讓每個字詞都有其重要性權重，刪除掉其中
