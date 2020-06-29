# Information Retrieval - Final Project Report

**Team 12**

**GitHub Repo**：https://github.com/jimpei8989/IR-Final-Project

**Team Name**：賈裴根陳尹症候群

| Name   | Student ID | Email Address               | Contribution                        | Signature |
| ------ | :--------- | --------------------------- | ----------------------------------- | --------- |
| 陳義榮 | B06902001  | <b06902001@csie.ntu.edu.tw> | Okapi、Doc2Vec、Doc2VecC            | <br>.     |
| 裴梧鈞 | B06902029  | <b06902029@csie.ntu.edu.tw> | Data Processing、Vector Space Model | <br/>.    |
| 賈本晧 | B06902039  | <b06902039@csie.ntu.edu.tw> | Report                              | <br/>.    |
| 尹聖翔 | B06902103  | <b06902103@csie.ntu.edu.tw> | BERT 相關部分                       | <br/>.    |

## 1. Introduction
我們想要探討 Deep Learning 的技術如何套用在 Information Retrieval 的領域上面，選用的方法包含 Doc2Vec、Doc2VecC、BERT，檢討其表現，並討論其與傳統的 vector space model 和 probabilistic model 等方法的差異，並做比較。

## 2. Related Works

### 2.1 Traditional Retrieval Models
在嘗試 deep learning model 前，我們先嘗試將課堂上介紹的 retrieval model 實作出來，並訂定 baseline。我們主要實作課堂上介紹的兩大經典 model，分別是 vector space model 以及 probability model。

### 2.2 Deep Learning Models
隨著 deep learning model 在各種不同的 domain 上表現得很好，如 BERT 在 textual 的 comprehensive tasks 上都表現得很好。我們也想嘗試將 BERT 等 model 套用在 information retrieval 的 task 上。

## 3. Methodology

### 3.1 Dataset
在這次的 project 中，我們使用 *TREC 2019* 中的 *Deep Learning Track* 作為實驗的 dataset。然而，這個 dataset 也的確造成了我們的一些麻煩：

1. Corpus 有 3,000,000 以上的 documents，這會造成 fitting、evaluation 的時候跑太久。<br>⇒ 我們只擷取其中 1,000,000 篇文章作為實驗的 corpus。
2. 文章中有許多非單字，如數學公式、網址等。這些造成 tokenizing 時出現大量少見的 token，或者 tokenizing 的成效不佳。

### 3.2 Vector Space Model
首先，我們使用的是最一開始教到的 vector space model。實作上，我們直接使用 `sklearn` 的 `TfidfVectorizer`。除了原本的 TFIDF 外，我們也嘗試套用一些變形：

- Sublinear TF：將 $tf$ 代換成 $1 + \log(tf)$。
- Feature selection：嘗試只取 title 作為 feature。
- Stemming：使用 `nltk` 的 `PorterStemmer`，tokenize 的部分也使用 `nltk` 的 `word_tokenize`。
- Relevance Feedback：使用 Pseudo Relevance Feedback 方式將每次 retrieval 的前 $k$ 名做 centroid。
- Latent Semantic Indexing：使用 SVD 將 tfidf 矩陣降維。

#### Challenges
1. 因為 corpus 有許多不常見的單字，如果全部使用的話會讓 feature 數量太多。因此，我們只選取
    - 該 token 出現次數小於 corpus 大小的一半
    - 該 token 至少出現在 100 篇不同的文章
    - 該 token 沒有在 english stopwords 中
2. 在這次的實作中，遇到的最大問題就是如果將 corpus load 進 memory，則會因記憶體不足而無法順利跑完。我們的解法是利用 offset，每次 seek 並讀取該片文章而已。原先以為這樣會讓 efficiency 掉很多，但結果是程式並沒有慢很多，可能 overhead 發生在其他地方。

### 3.3 Probabilistic Model
我們使用 Okapi probabilistic model，並加上了 relevance feedback 的技術。對於每個 query $Q=q_1q_2\dots q_n$ 與 任意一個 document $D$ 的 BM25 score 為：

$$
\operatorname{score}(D, Q)=\sum_{i=1}^{n} \operatorname{IDF}\left(q_{i}\right) \cdot \frac{f\left(q_{i}, D\right) \cdot\left(k_{1}+1\right)}{f\left(q_{i}, D\right)+k_{1} \cdot\left(1-b+b \cdot \frac{|D|}{\operatorname{avgdl}}\right)},
$$

其中 $f(q_i, D)$ 為 $q_i$ 在 $D$ 的 term frequency，$\vert D\vert$ 為 $D$ 的長度，$\text{avgdl}$ 為所有 document 的平均長度，$IDF(q_i)$ 為 $q_i$ 的 inverse document frequency。在此次實驗中，我們令 $b=0.75$，我們會嘗試不同的 $k_1$ 與 relevance feedback iteration 次數。

#### Implementation
我們使用 `sklearn` 的 `TfidfVectorizer` 來獲得原始的 TF 與 IDF，再轉換成 Okapi 的樣子。對於每個 query，我們算出它與每個 document 的評分後再取前 1,000 名。

#### Relevance feedback
Relevance feedback 採用 predict 出來的前 100 名 document 當作 pseudo relevant set $D_r$，將他們的 document vector 與原來的 query vector 做線性組合。

$$
\hat{q}_i=\alpha q_i + \frac{(1-\alpha_i)}{\vert D_r \vert} \sum_{d_i \in D_r} d_i
$$

#### Challenges
在存取 TF-IDF 時，因為整個矩陣過於巨大，所以得用 sparse matrix 來儲存；又因為計算 1,000,000 筆 document 和所有的 query 的 score 會花太多時間，所以我們使用 thread pool 來平行化。

### 3.4 Doc2Vec, Doc2VecC
我們嘗試了 Doc2Vec 與 Doc2VecC。Doc2Vec 是使用 Paragraph Vector - Distributed Memory (PV-DM) 的實作方法。對於每個 query $Q$，算出其 document vector 後再跟其他所有的 document vector 算 cosine simularity。

#### Implementation
* Doc2Vec 使用 `gensim` 的 `Doc2Vec`，vector size = 100
* Doc2VecC 使用 [Doc2VecC_python](https://github.com/taikamurmeli/Doc2VecC_python) 的 `Doc2VecC`，vector size = 128

#### Relevance feedback
Relevance feedback 採用 predict 出來的前 100 名 document 當作 pseudo relevant set $D_r$，將他們的 document vector 與原來的 query vector 做線性組合。

$$
\hat{q}_i=\alpha q_i + \frac{(1-\alpha_i)}{\vert D_r \vert} \sum_{d_i \in D_r} d_i
$$

#### Challenges
Doc2Vec、Doc2VecC 雖然占用的記憶體不大 $(1,000,000 \times 128 \times 4 = 512 \text{ MB})$，但如果要一起計算 50,000 筆 query 和 1,000,000 筆 training data，算出來的 cosine simularity matrix 會非常大 $(50000 \times 1,000,000 \times 4 = 200\text{ GB})$。所以我將 query 切成一個一個 batch，每次計算一個 batch 和 training data 的 cosine simularity。

### 3.5 BERT
我們使用 BERT 作為 document 及 query 的 encoder，並以一層的 linear classifier 計算出兩者的 relevant score。

#### Implementation
我們採用 Huggingface 的 `transformers` 套件中提供的 BERT model，tokenizer 以及 model pretrained weight 的部分皆使用 `bert-base-uncased`。我們以 BERT output 中對應 CLS (段落開頭)的部分作為 embedding，再將 document 和 query 的 embedding concatenate 起來輸入 linear classifier，最終輸出一個數字作為該 document 和 query 的 relevant score。
在訓練的過程中，我們並沒有 fine-tune BERT 所有的層數，而是隨著 epoch 變大增加 freeze 住的層數並且放大 batch size。我們使用 4 倍的 negative sample 來生成 training set，Optimizer 採用 pytorch 提供的 AdamW，Loss 的部分則採用 BCELoss。

* Model Architecture
    ```python
    # Tokenizer from pretrained: 'bert-base-uncased'
    # BERT from pretrained: 'bert-base-uncased'
    
    documentEmbedding = BERT(Tokenizer(document))[0]
    queryEmbedding = BERT(Tokenizer(query))[0]
    
    Score = Fully-Connected( concat(documentEmbedding,queryEmbedding) )
    ```

#### Challenges
* 因為 data 的量太大，必須有效率的取樣來生成 training dataset，而訓練的過程也非常地耗時。除此之外，因為預測的過程必須對每一個 query 都算出和所有 document 的 relevant score，若使用和訓練時一樣的方法，則運算量會非常的可怕。我們必須先算出每一個 document 和 query 的 embedding 再分別計算出其對應分數，而最後對所有分數的 sort 也非常的耗時。
* 因為許多 document 的內文在 tokenize 後都超出 BERT 可以接受的長度，造成許多內容最終被刪掉，並未納入訓練過程，我們猜測這可能是導致 BERT 的效果不佳的原因之一。


## 4. Experiments
### 4.1 Experiment Results
<!-- → Fill in this [Google Sheet](https://docs.google.com/spreadsheets/d/1dmGR1l_o38IhftNr8FNHT7YXvhinaeucHMb2PGTH5kQ/edit?usp=sharing) first. I'll make the table later. -->

|  Model   | Description                                                  | Training MAP | Development MAP | Testing MAP |
| :------: | ------------------------------------------------------------ | :----------: | :-------------: | :---------: |
|   VSM    | No stemming, linear TF                                       |    0.2185    |     0.2122      |   0.1977    |
|   VSM    | No stemming, sublinear TF                                    |    0.1660    |     0.1616      |   0.1484    |
|   VSM    | No stemming, linear TF, use Title as features only           |    0.0392    |     0.0376      |   0.0418    |
|   VSM    | Stemming, linear TF                                          |    0.1629    |     0.1592      |   0.1468    |
|   VSM    | No stemming, linear TF, relevance feedback (1 iteration, $\alpha=0.95$) |    0.2182    |     0.2118      |   0.1970    |
|   VSM    | No stemming, linear TF, relevance feedback (1 iteration, $\alpha=0.8$) |    0.1761    |     0.1713      |   0.1565    |
|   VSM    | No stemming, linear TF, LSI (rank=128)                       |    0.0120    |     0.0114      |   0.0110    |
|   VSM    | No stemming, linear TF, LSI (rank=256)                       |    0.0230    |     0.0204      |   0.0220    |
|   VSM    | No stemming, linear TF, LSI (rank=512)                       |    0.0380    |     0.0270      |   0.0366    |
| Doc2VecC | No relevance feedback                                        |    0.0143    |     0.0134      |   0.0128    |
| Doc2VecC | Relevance feedback (2 iterations, $\alpha=0.8$)              |    0.0144    |     0.0134      |   0.0130    |
| Doc2Vec  | No relevance feedback                                        |    0.0026    |     0.0026      |   0.0025    |
| Doc2Vec  | Relevance feedback (2 iterations, $\alpha=0.8$)              |    0.0025    |     0.0023      |   0.0025    |
|  Okapi   | $k_1=1.2$, no relevance feedback                             |    0.1181    |     0.1138      |   0.1048    |
|  Okapi   | $k_1=1.2$, relevance feedback (2 iterations, $\alpha=0.8$)   |  **0.2307**  |   **0.2188**    | **0.1977**  |
|  Okapi   | $k_1=1.5$, relevance feedback (2 iterations, $\alpha=0.8$)   |  **0.2307**  |   **0.2188**    | **0.1977**  |
|  Okapi   | $k_1=1.5$, relevance feedback (3 iterations, $\alpha=0.8$)   |    0.1625    |     0.1545      |   0.1390    |
|   BERT   | bert-base-uncased, learning rate $10^{-4}$, 6 epochs         |    0.0327    |     0.0316      |   0.0298    |

### 4.2 Findings

##### VSM
1. 最基本的 TFIDF model 就有不錯的表現。
2. Stemming 並沒有將 performance 提升。我們猜測是因為 stemming 造成上課提到的 "hurt precision, enhance recall" 的現象。此外，stemming 的效率不高，檢索的時間會增大數倍，整體而言對於 retrieval model 傷害太大。
3. Pseudo relevance feedback 也沒有讓 performance 提升，
4. Lantent Semantic Indexing (LSI) 的 performance 也沒有很好，我們猜測是 rank 取的不夠多而導致的。然而，計算 SVD 也需要時間，我們就沒有再嘗試提高 rank 的大小了。

##### Okapi
1. 最基本的 Okapi probabilistic model 表現沒有 VSM 好。
2. 加了 Pseudo relevance feedback 後，表現明顯提升不少。
3. 更改 $k_1$ 的值沒有對 performance 造成影響。
4. 進一步增加 feedback 的次數，發現表現不增反減，我們猜測是 pseudo positive items 裡可能有誤導 model 的項目，導致表現降低。

##### Doc2Vec, Doc2VecC
1. Doc2VecC 的結果比 Doc2Vec 好上不少。
2. 加了 relevance feedback 後，Doc2VecC有些微提升，但 Doc2Vec 則沒有明顯差異。

##### BERT
1. 在訓練過程中，在第一個 epoch 結束前就可以從 loss 和 predict accuracy 看出已經收斂。我們猜測原因是我們採用的 BERT model 加上 linear classifier 並不需要太多 data，過多的 data 反而會造成 overfit。
2. 在 overfit 之後，可以看到 validation accuracy 保持在 0.8，代表在分類是否相關時有 80% 的準確率。而這個數字的來源是因為我們 negative sample 的數量是 positive 的 4 倍，也就是 model 不管輸入一概將之分為不相關就會有 80% 的準確率。若是時間充裕，我們應該嘗試使用 BCEWithLogitsLoss 來取代 BCELoss，以解決 positive 和 negative sample 不平均的問題。

## 5. Conclusions
在這次的實驗中，我們發現傳統的方法不一定就會比近代的 deep learning 的方法差，可見傳統的方法（VSM、probabilistic model）仍然有其一定的競爭力。另外，我們認為 relevance feedback 扮演了非常重要的角色，fine-tune relevance feedback 的參數會對結果造成明顯的差異。

## 6. References
* TREC 2019 - Deep Learning Track: <https://microsoft.github.io/TREC-2019-Deep-Learning/>
* Document Embedding Techniques: <https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d>
* Modern Information Retrieval: A Brief Overview: <http://singhal.info/ieee2001.pdf>
* A Study of Smoothing Methods for Language Models Applied to Ad Hoc Information Retrieval: <https://dl.acm.org/doi/pdf/10.1145/3130348.3130377>
* Doc2Vec: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
* Doc2Vec paper: Le, Q., & Mikolov, T. (2014, January). Distributed representations of sentences and documents. In International conference on machine learning (pp. 1188-1196).
* Doc2VecC: https://github.com/taikamurmeli/Doc2VecC_python
* Doc2VecC paper: Chen, M. (2017). Efficient vector representation for documents through corruption. arXiv preprint arXiv:1707.02377.
* Okapi: https://en.wikipedia.org/wiki/Okapi_BM25
* BERT: https://huggingface.co/transformers/
* BERT paper: Dai, Z., & Callan, J. (2019, July). Deeper text understanding for IR with contextual neural language modeling. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 985-988).