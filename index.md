## Parse Out Email Text

>1.讀取email文件，並從中擷取出email的內容
>
>2.去除掉email內容中的標點符號和空格
>
>3.對email的內容，取出每個英文字的Stem

#### 引入所需的Library

```python
from nltk.stem.snowball import SnowballStemmer
import string
```

#### 從檔案f讀取email文件

```python
### go back to the beginning of the file and read all the text
def parseOutText(f):
    f.seek(0)
    all_text = f.read()
```

以下讀取一個email的範例：

```
To: Katie_and_Sebastians_Excellent_Students@udacity.com
From: katie@udacity.com
X-FileName:

Hi Everyone!  If you can read this message, you're properly using parseOutText.  Please proceed to the next part of the project!
```

#### 擷取出email文件中的內容

從email文件範例中可以發現，email的內容存在於 **X-FileName:** 之後，因此以 **X-FileName:** 作分割， **content[1]** 即為email的內容

```python
### split off metadata
content = all_text.split("X-FileName:")
```

擷取出的email內容， **content[1]** :

```


Hi Everyone!  If you can read this message, you're properly using parseOutText.  Please proceed to the next part of the project!
```

#### 去除掉email內容中的標點符號，換行符，Tab鍵

```python
### remove punctuation
text_string = content[1].translate(str.maketrans("", "", string.punctuation))
### replace \n, \t, \r with white space
for i in ['\n', '\t', '\r']:
    text_string = text_string.replace(i, " ")
```

處理過後的email內容：

```
  Hi Everyone  If you can read this message youre properly using parseOutText  Please proceed to the next part of the project
```

#### 對每個英文字進行Stem轉換

例如英文字中的response, responsive, responsibility, responsible，經過Stem轉換後皆為 **respons** ，在統計文件中文字個數時，這些具有相同Stem的英文字皆被算成同一種字

```python
### split the text string into individual words, stem each word,
### and append the stemmed word to words (make sure there's a single
### space between each stemmed word)
stemmer = SnowballStemmer('english')
text_list = text_string.strip().split(' ')
words = ''
for word in text_list:
    if len(word):
        words = words + stemmer.stem(word) + ' '
```

經過Stem轉換後的email內容：

```
hi everyon if you can read this messag your proper use parseouttext pleas proceed to the next part of the project
```

## Pickle Email Text

>1.讀取紀錄Sara和Chris的Email文件路徑
>
>2.經由ParseOutText函式將Email的內容去除掉標點符號和進行Stem轉換
>
>3.使用Pickle儲存經過處理後的Email內容

#### 引入所需的Library

函式parseOutText的內容，請見於上一章 **Parse Out Email Text** 的解釋

```python
import os
import pickle

from parse_out_email_text import parseOutText
```

#### 讀取紀錄Sara和Chris的Email文件路徑

./maildir資料夾下儲存了Enron DataSet，其內存在海量的Email文件，因此需要從中找尋出只屬於Sara和Chris的Email文件

**from_sara.txt** , **from_chris.txt** 紀錄了在Enron Dataset當種只屬於Sara和Chris的Email文件路徑，例如：

```
###擷取自from_sara.txt

maildir/bailey-s/deleted_items/101.
maildir/bailey-s/deleted_items/106.
maildir/bailey-s/deleted_items/132.
maildir/bailey-s/deleted_items/185.
.
.
.
```

因為 **pickle_email_text.py** 存在於 **./preprocess** 資料夾當中，所以需要在Sara和Chris的Email文件路徑的前面加上../，並將其儲存於path當中

```python
from_sara = open('from_sara.txt', 'r')
from_chris = open('from_chris.txt', 'r')

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
	for path in from_person:	
		path = os.path.join('..', path[:-1])
		email = open(path, 'r')
```

其結果為：

```
###擷取自from_sara.txt路徑被轉換後的結果

../maildir/bailey-s/deleted_items/101.
../maildir/bailey-s/deleted_items/106.
../maildir/bailey-s/deleted_items/132.
../maildir/bailey-s/deleted_items/185.
.
.
.
```

#### 經由ParseOutText函式將Email的內容去除掉標點符號和進行Stem轉換

```python
text_string = parseOutText(email)
```

#### 使用Pickle儲存經過處理後的Email內容

將Email內容以list的方式儲存於 **word_data** ，list當中的每一個Index對應一封Email

同時將Email的作者以list的方式儲存於 **email_authors** ，屬於Sara的Email計為0，屬於Chris則計為1

```python
### append the text to word_data
word_data.append(text_string)
### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
if name == "sara":
	email_authors.append(0)
else:
	email_authors.append(1)
```

使用Pickle儲存 **word_data** 和 **email_authors**

```python
pickle.dump(word_data, open('word_data.pkl', 'wb'))
pickle.dump(email_authors, open('email_authors.pkl', 'wb'))
```
