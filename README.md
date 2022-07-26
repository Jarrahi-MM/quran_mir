پروژه Quran_Mir به عنوان پروژه‌ی درس بازیابی پیشرفته اطلاعات در بهار ۱۴۰۱ انجام شده‌است. هدف از انجام این پروژه، ایجاد یک موتور جست‌و‌جو برای آیات قرآن بوده است. همچنین به کمک روش‌های آماری و الگوریتمی، ابزارهای دیگری مانند تشخیص آیات محوری، خوشه‌بندی آیات به صورت مفهومی به ۲ دسته (که با دقت ۹۰ درصد، معادل دسته‌بندی مکی/مدنی شد)، و … توسعه داده شده است.<br />
از آنجا که توسعه و ارزیابی مدل‌ها و ذخیره‌ی نتایج آن‌ها در مقایسه با توسعه‌ی وبسایت و نمایش خروجی‌ها محیط کاملا متفاوتی نیاز دارد، این پروژه در ۲ مخزن ذخیره شده‌است.<br />
- مخزن اول در آدرس https://github.com/Jarrahi-MM/quran_mir حاوی بخش علمی پروژه است. تمام کد‌ها و نتایج و ارزیابی مدل‌های مختلف در این مخزن قرار دارد. <br />
- مخزین دوم در آدرس https://github.com/IR1401-Spring-Final-Projects/Quran1401-1_20 حاوی بخش وبسایت پروژه است. برخی کدها به صورت مستقیم از مخزن دیگر در این مخزن قرار گرفته‌اند و برای کدهای دیگر، صرفا خروجی‌های مدل‌ها آورده شده‌اند. برای برخی کدها نیز صرفا نتایج بررسی آیات و سوره‌ها در قالب فایل‌های اکسل در این مخزن قرار گرفته است و صرفا از آن‌ها استفاده می‌شود. <br />
در هر کدام از مخزن‌ها، در فایل‌های README.md به توضیح ساختار مخزن پرداخته شده‌است. <br />

### tools.py
Functions such as creating random dataset for fasttext training and computing similarity between query and documents.

### search.ipynb
Examples of using different methods of IR(fasttext, arabert, boolean, tfidf) & query expansion module.

### Query_expansion.py
This file gets a query as input & expands that query based on rocchio algorithm.

### preprocess_quran_text.py
This file is used for preprocessing on quran text, quran lemma text & quran root text based on customized parsi.io librabry.

### MRR_Comparsion
This file includes 4 csvs that compare IR results on expanded queries & original queries.

### boolean_retrieval
This folder contains boolean system to parse queries with 'AND' 'OR' 'NOT' operators.


## Different IR Classes
There are different classes in quran_ir.py file used for IR.
### BooleanQuranIR Class
This class is used for IR based on boolean method.
### TfIdfQuranIR Class
This class is used for IR based on tfidf method.
### FasttextQuranIR Class
This class is used for IR based on fasttext method.
### ArabertQuranIR Class
This class is used for IR based on transformer method.
### QuranIR Class
All above classes inherit this class, to get search results for queries.

## Query files
### queries.txt 
This file is used for evaluating Elastic Search & fasttext IR, Arabert IR, tf-idf IR.
### queries_boolean.txt 
This file is used for evaluating boolean IR.


## Clustering & Classification files
### transformer_classification.ipynb
In this notebook we use Arabert and fine-tune it in order to use it for our classification task.

### classification.py & classification.ipynb
This file uses MLP_classifier for fasttext embeddings to predict sura of each aye of long suras.
Also uses Logistic regression for tfidf embedding to do the same work.

### clustering_w2v.ipynb
This file clusters suras into makki & madani, & it also clusters all ayes into 4 different semantic parts based on w2v embeddings. 

### clustering.py & clustering.ipynb
This file clusters suras into makki & madani based on fasttext embeddings. 

### csv_outputs
This file contains csv outputs of clustering & classification tasks.

## Link Analysis files
### quran_ranker.py & quran_ranker.ipynb
In this file, we use networkx library to find pivot aye for each sura.
### linkAnalysis.ipynb
This notebook contains pivot aye for each sura based on Hits alg. & pagerank alg & comparison of the results.
