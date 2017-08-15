## Parse out email text

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
