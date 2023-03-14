# Emotion-detection-using-Tweets-and-Naive-Bayes



Thamarai Kannan P , Yuva Rani V H , Nirmala R PG scholar, SCOPE, VIT University, Vellore, India



The Problem domain is Social Media Analytics which is refers to collection of vast amount of data from various social media platforms, and also used to analyze it further. Sentiment analysis is the process of understanding an opinion about a given data. It can performed on written language for example emails, chats, tweets, comments etc..,. As title indicates Our project is performing Sentiment analysis on Twitter data. It involves many steps like to Gathering twitter data, Creating a Sentiment Analysis model , Classification of data using  Naïve Bayes classification and  Visualizing the results. The Application area is use the analytical tool for example, they use it for the movie reviews, celebrity popularity, and for the advertisement of any products and getting positive and negative reviews about it. In Existing works the sentiment analysis on twitter was mostly used for emotion detection from chats and for political or business ratings and reviews the Polling methods, and Star ratings are basically used  to see the response from people. In our Proposed work, we are going to use Twitter API along with Tweepy python package to fetch the data and Naïve bayes approach for classification. The tweets of user along with retweets and comments can be collected , processed and the positive, negative and neutral response for tweets from the people can be visualized. The information obtained from this system can be used in various applications like Analysis of social media support for politician, A review based on user response for a Movie or Product, Response from people for social or political issues (Hashtags), Response for Government announcements etc..,. The system will helps to visualize statistics, to analyse people response and provide most Effective and Useful statistical tool for various Industries.


Social media platforms are one of the most used platforms by people. Enormous population using it more often nowadays to share their thoughts fearlessly regarding various political , social , economical and business issues. Social Media Analytics (SMA) refers to process of analyzing data in social media platforms .It involves collection of vast range of data from various media platforms Like Facebook ,Twitter, Instagram sites etc…,. which are then preprocessed , analyzed and visualized by applying various NLP and ML algorithms on it, and this paradigm usually used to build various analytical tools. It gathers analysed data of people and visualize them as useful information. It also helps to understand and analyze the unique data and also shows how well the social media is performing in day-to-day life. 

Sentiment analysis is the process of understanding an view about a given subject of an user. Sentiment analysis can also be known as Opinion mining . There are different types of sentiment analysis which are Aspect based analysis, Fine grained analysis models, Intent analysis Emotion based model etc..,. In our project we are going to implement Twitter sentiment analysis based on Emotion that enable us to know about what's being said by people about any product , movie, politician or celebrity, and the retweets being positive, negative and neutral responses. And that helps the people to know about it more. The datasets from Twitter are collected using an Twitter API which is a tool provided by twitter for accessing data. Along with it a Tweepy python package is also used. The Pre-processing collected texts can be done by various NLP methods like stop word removal , Tokenization etc..,. and Finally to classify data based on emotions Naïve Bayes classification approach is used here, which is known as strong assumption of the features. 

The Naive Bayes classification is an Supervised Machine Learning algorithm is a probabilistic classifier. This algorithm works based on Bayes  theorem . It is the popular baseline method category for the texts and judging it the features of it. The main advantage of this is, it requires only a small amount of data for the classification. This could correctly classify users data of the given datasets as positive , negative or neutral responses by using Bayes probabilistic model. The model is build by training various datasets with their probability values and building a strong Bayesian model. The Input datasets are compared with Model and classified accordingly as positive , negative or neutral based on probability values. 



This system acts as tool for various statistical approaches. It can be used by Media to view twitter statistics of particular event or news ,Organization can use this for response from people ,Movie industries can visualize the view on their movies , A response from people for any Government schemes etc..,. can be Analysed by implementing Sentiment analysis on Twitter.


Referring to sources in general, current research on Popularity based opinion mining on twitter data .Previous studies indicates that the sentiment analysis on twitter was mostly used for emotion detection from chats. 

The literature [1] shows a variety approaches they have done using Machine learning .The main purpose of this study is to emotion of citizens from different people’s sentiment.They used Deep long short-term memory (LSTM) models and used for  sentiment polarity and emotions from fetched tweets. They used  unique way of gathering the supervised deep learning models on tweets extracted from Twitter.[1] they used and fetched the tweet datasets are used for detecting both sentiment polarity and emotions from users’ tweets on Twitter. Proposed a multi-layer LSTM assessment model for classifying both sentiment polarity and emotions. The disadvantage of this paper is in some point  the feelings and emotions can not  be detected correctly . Shadi Shaheen et al [2] where they classified the emotions and extracted the concepts from sentences and given the medium emotional data which represent the given input sentence by the semantic structure and syntactic model. Then they globally  represented  by using Word-net and Concept-net which gives the result as ERR emotion recognition .They used classifiers to compare the emotion recognition and also the (KNN) k-nearest neighbours with the similarity measures. The disadvantage of this paper is they used different datasets, the approach in working model showed the learning and rule based classifiers with the score of 84%average. [3] Shihab Elbagir  and Jing Yang proposed the method to fetch the tweet's by efficient feature. They used  multinomial logistic regression, SVM, Decision Tree and also the Random Forest to classify the sentiment analysis. The work study of this paper can detect ordinal regression with good correctness results by use of machine learning. The main advantage od this study is  using Decision tree it gives the best results compared to other algorithms. Even though, they compared with four algorithms . the accuracy they get is not good enough. Decision tree alone gives the accurate result. But the thing which decision tree gives the discriminative model compared to Naive bayes where it is generative one.


According to  Bouazizi and Ohtsuki [4]  they used senta, that helps users to select the more ranges of features that suite the application which runs the classification. Their  work study is to analyse  the text from tweets with multi-class sentiment, but their method is limits to seven different sentiments. The proposed framework and the results shows it reached an accuracy as high 60.2%. This method from their study gives sufficient accurate with binary and ternary classification. But compared to this paper other algorithms fives more accuracy. Shiyang Liaoa et al [5]  Analysed twitter data and analysing the classification of the global information. By giving and using the neural network they can able to get higher accuracy of their classification. Because they natural language processing, and they fetch text and tweets  and also extract the whole content and the sentences. The CNN is most effective one in nowadays as it is the advantage now, they do effective in image classification. But in present days there are more works done than this, [5] their study and proposed framework is twitter data by sentiment in that they constructed CNN based on text, but we can do more than text. And another study was done [6] that they used classification method in twitter by English text data. Where this study approach was to detect the hate speech which is addressed to black people[7] As well, the study work related to detection of hate speech which is used by Indonesian language is very few, and it is related to  the politic they done. By gathering the data, which are used is hate speech and the keywords to some election in 2017. The study work focused on Random Forest, decision Tree also known as the best model algorithm which gives the Measure compared to other models. But the disadvantage of their accuracy and detection religion from text alone cant help[8]


According to T T A Putri , S Sriadhi et al  [9] the application model used classification models with the machine learning such as Multi level Perceptron, Naive Bayes , decision tree and support vector machine. Their study compared with the performance of by using SMOTE to get the imbalanced data. But at last the comparative study shows the Multinomial Naive Bayes algorithms gives the accurate and good data withe best accuracy. Based on this paper, they discovered SMOTE is not affecting much in the classification model, but by seeing this the MLP algorithm gives the accurate value by scarping the tweets of hate speech. After comparing with all other algorithms all uni-gram without SMOTE. As the conclusion, we suggest Multinomial Naïve Bayes algorithm by using unigram feature without SMOTE as the best model to classify a hate speech. 


3. PROPOSED MODEL:

In summary by applying and studying all the previous work says and compared many algorithms and shows which is the best, but   work, we are going to use Twitter API along with Tweepy python package to fetch the data and Naïve bayes approach for classification. The tweets of user along with re-tweets and comments can be collected , processed and the positive, negative and neutral response for tweets from the people can be visualized. The information obtained from this system can be used in various applications like Analysis of social media support for political predictions , product reviews and social support etc..,.

A)Data Collection:

In this study we collected data by scraping the tweets, and we collected the twitter data by using Twitter API(application program Interface).We used python programming language for the usage of Twitter API. We created Twitter API account . In Tweepy API we created one new account which is linked to our twitter account which can access Twitter API. By Using code we fetch  the Tweets which we are searching for keywords  Based  on those keywords, this data collection  is not only about covid pandemic but also about the public opinion. We collected data as many tweets and the data that we used in this research. And atlast , we identify the tweets which consists of positive , negative and neutral ones. We will show the result by comparing it.


B)Pre-processing of Tweets:

Some of the tweets which are full of incorrect , noisy , and numerous abbreviations and slang words. Such tweets are often involved in the performance of sentimental analysis. There are some pre-processing methods which are applied to proceed further. The followings steps for pre-processing are:

Tweets are pre-processed by normalizing extra letters such as (“worlld” will become “world”) and misspelled words like (“bcoz” to “because”). 
 Removing all hashtags like “# Hashtags” and hyperlinks will take us to unwanted page “(url:/http:)” that are in tweets.
 Avoiding unwanted spaces and special characters or symbols such as “ < > , ‘ ? ; : ! @ # $ % & * ~ / \ [ ] { } + = _ - ( ) | ^ . 
 Removing stop words such as “all, is, was, that, this, of, so, such, etc..” will not add more meaning to the sentence.
 Avoiding duplicated tweets and emojis.
Removing case conversions.
 Separating sentences into phrases or words are said to be tokens by which some characters are discarded (Tokenization)


C)CLASSIFICATION:

To form a result from feature of extraction method classification process is done by using the data. This process is done by using machine learning and some tools for classifying the sentiment analysis and then it is used by the mining classification as the method to predict the sentiment anlayser of tweets. The result of this study is processed by classification model of using Navie Bayes algorithm.

Naive Bayes classifiers are linear classifiers and this model is generated from probabilistic. This algorithms estimates the conditional probability of somethings that will happen and some other thing that is already occurred. The idea which we use here in this study that can give more accurate probability for prediction. The main advantage of this is, it requires only a small amount to fetch the tweets of  classification. This could correctly classify users data of the given datasets as positive , negative or neutral responses by using Bayes probabilistic model which is shown in Figure 2 . It is fast and easy to predict the data and also perform  good in many class prediction.In this research that becomes test data is document tweets.





Figure 2 Naïve bayes model



 Naïve Bayes Classifier (With Training Set Scheme) 

 Step 1: Create data files for the classifier. 
Create a file of tweets with their sentiment of each sentiment analyzer (test set). 
Create a file of negative and positive labeled tweets of each sentiment analyzer (training set). 
Convert all csv files to art format file.

 Step 2: Build Naïve Bayes classifier model 
Create model of each analyzer by providing the training set file. 

 Step 3: Execution of model on test set. 
Load the test set file. 
Apply the String To Word Vector  filter with following parameters: 
IDF Transform: true, TF Transform: true, stemmer: Snowball Stemmer, Stop word handler: rainbow, tokenizer: Word Tokenizer. 
Execute the model on the test set. 
Save results in the output file

Multinominal Model: It is the Naive Bayes algorithm and the probabilistic method which is used in Natural language Processing(NLP). This model is based on Bayes theorem which predicts the text of tweets. And also t features the vector where the given of characteristics represents as many times it appears for the distribution of these features.


Tools and Libraries: Python scripts are used to query Tweepy Twitter API1 for fetching users’ tweets and extracting feature set for Using Sentiment Analysis , NLTK is used to preprocess the retrieved tweets. instrument to analyze the results in addition to correlation are used. We will use the wonders of the awesome data wrangling library known as Python Scikit-Learn; Numpy; Seaborn; NLTK; Wordcloud  We get our data in the form of JSON, then we perform preprocessing and ultimately combine all comments followed by the sentiment score (using TextBlob) into a DataFrame.


The model was implemented based on which the results are obtained as visual and mathematical values. The dataset from hashtag of COVID is taken for processing using an Twitter API. The had been converted into an dataframe and used to create an training and testing models. 


The confusion matrix visualize the difference between the number of tweets analysed and predicted . This matrix value gives the number of positive , negative and neutral as well as false positive , negative and neutral. 



The Training dataset with pre defined polarity values for each sentence is used to Build a Prediction model using which the Test datasets are loaded and predicted for sentiments. The outcomes are obtained graphically as pie chart and bar charts which are represented as positive , negative and neutral sentiments.


The obtained Naïve bayes prediction model is tested for accuracy score in which the predicted values are compared with actual values of dataset and the Accuracy score is obtained.

This paper focused on opinion mining of twitter data to obtain a positive , negative and neutral popularity of tweets. The data are gathered , pre-processed and classified based on their sentiments using Naïve bayes classifier . The prediction model is built using a Training dataset and the visual representations are also given. The Accuracy of obtained results are verified by comparing the predicted data with actual data where percentage oscillates from 80% to 90% according to given data and the average accuracy is 84 %  which was shown in Figure. 9. The percentage of accuracy is quite low compared to other models such as SVM , ANN etc..,. But the speed of processing and the ease of implementation is high for Naïve bayes model. Also the model gives high accurate prediction with small training dataset. The future work of this model is to build it above an web application to make an online analytical tool that visualize the opinions of tweets that comes under particular hashtag or user.  



REFERENCES:


[1]Cross-Cultural Polarity and Emotion Detection Using Sentiment Analysis and Deep Learning on COVID-19 Related Tweets Ali Shariq Imran 1 , (Member, IEEE), Sher Muhammad Daudpota 2 , Zenun Kastrati 3 , and Rakhi Batra 2

[2]Shadi Shaheen, Wassim El-Hajj et.al, “Emotion Recognition from Text Based on Automatically Generated Rules”, in 2014 IEEE International Conference on Data Mining Workshop

[3]Twitter Sentment analysis Based on Ordinal Regression SHIHAB ELBAGIR 1,2 AND JING YANG 1College of Computer Science and Technology, Harbin Engineering University, Harbin 150001, China 2Faculty of Computer Science and Information Technology, Shendi University, Shendi 142-143, Sudan

[4]M. Bouazizi and T. Ohtsuki, ‘‘A pattern-based approach for multi-class sentiment analysis in Twitter’’ IEEE Access, vol. 5, pp. 20617–20639, 2017.

[5]CNN for situations understanding based on sentiment analysis of twitter data Shiyang Liaoa, b, Junbo Wangb , Ruiyun Yu, Koichi Sato, Zixue Cheng

[6]Kwok I and Wang Y 2013 Locate the Hate: Detecting Tweets against Blacks. Proceedings of the Twenty-Seventh AAAI Conference on Artificial Intelligence (pp. 1621-1622). Association for the Advancement of Artificial Intelligence 

[7]Mulia R, Alfina I, Fanany M I, and Ekanata Y 2017 Hate Speech Detection in the Indonesian Language: A Dataset and Preliminary Study The 9th International Conference on Advanced Computer Science and Information Systems (ICACSIS 2017) 

[8]Pratiwi S H 2016 Detection of Hate Speech against Religion on Tweet in the Indonesian Language Using Naïve Bayes Algorithm and Support Vector Machine (Universitas Indonesia) 

[9]A comparison of classification algorithms for hate speech detection T T A Putri , S Sriadhi, R D Sari, R Rahmadani, and H D Hutahaean PTIK-FT, Universitas Negeri Medan, Indonesia 

[10]Emotion Detection of Tweets using Naïve Bayes Classifier Hema Krishnan Research scholar, School of Engineering, CUSAT M. Sudheep Elayidom Associate Professor, School of Engineering, CUSAT T. Santhanakrishnan Scientist E, NPOL

[11]A Novel Method for Twitter Sentiment Analysis Based on Attentional-Graph Neural Network Mingda Wang  and Guangmin Hu School of Information and Communication Engineering, University of Electronic Science and Technology of China, Chengdu 611731, China; hgm@uestc.edu.cn 

[12]Twitter Sentiment Analysis: A Bootstrap Ensemble Framework Ammar Hassan and Ahmed Abbasi Department of Systems and Information Engineering  Department of Information Technology University of Virginia Charlottesville, Virginia USA mah9tg@virginia.edu, abbasi@comm.virginia.edu Daniel Zeng Department of Information Systems ,University of Arizona, Tucson, Arizona, USA Institute for Automation, Chinese Academy of Sciences Beijing, China zeng@email.arizona.edu 

[13]Topic Sentiment Analysis in Twitter: A Graph-based Hashtag Sentiment Classifification Approach Xiaolong Wang  , Furu Wei  , Xiaohua Liu  , Ming Zhou , Ming Zhang  School of EECS, Peking University, Beijing, China  Microsoft Research Asia, Beijing, China  xwang95@illinois.edu, mzhang@net.pku.edu.cn .

[14]Comparison Research on Text Pre-processing Methods on Twitter Sentiment Analysis ZHAO JIANQIANG1,2,3 and GUI XIAOLIN1,3 1School of Electronic and Information Engineering, Xi’an Jiaotong University, Xi’an 710049, China 2Xi’an Politics Institute, Xi’an, Shaanxi Province, 710068, China 3Key laboratory of Computer Network of Shaanxi Province, Xi’an 710049, China

[15]Sentiment Analysis on Twitter Data Varsha Sahayak Vijaya Shete Apashabi Pathan BE (IT) BE (IT) ME (Computer) Department of Information Technology, Savitribai Phule Pune University, Pune, India. 

[16]Gender Detection of Twitter Users based on Multiple Information Sources Marco Vicente1,2 , Fernando Batista1,2 , and Joao P. Carvalho1,3 1L2F – Spoken Language Systems Laboratory, INESC-ID Lisboa 2 Instituto Universit´ario de Lisboa (ISCTE-IUL), Lisboa, Portugal 3 Instituto Superior T´ecnico, Universidade de Lisboa, Portugal.

[17]Emotion Detection and Analysis on Social Media1 Bharat Gaind CSE Department IIT Roorkee Roorkee, India bharatgaind234@gmail.com Varun Syal CSE Department IIT Roorkee Roorkee, India varunsyal1994@gmail.com Sneha Padgalwar CSE Department IIT Roorkee Roorkee, India sneha.padgalwar@gmail.com.




