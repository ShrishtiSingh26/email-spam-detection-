# email-spam-detection-
In today’s digital age, email remains a fundamental tool for communication. However, with the convenience of email comes the persistent problem of spam. Spam emails clutter our inboxes, wasting our time and potentially exposing us to various scams and phishing attempts. To combat this issue, machine learning (ML) models have emerged as a powerful solution for classifying emails into two categories: “spam” or “ham” (non-spam).

Spam emails are unsolicited, irrelevant, or malicious messages that flood our email inboxes, often with the intention of deceiving or defrauding the recipient. Ham, on the other hand, comprises legitimate emails.
The goal is to accurately identify spam emails to prevent them from reaching users' inboxes, thus improving email security and user experience.
Clean and preprocess the emails by removing stop words, stemming, and converting text into numerical representations was done using TF-IDF vectorizer.

![security-email_spam_mobile](https://github.com/ShrishtiSingh26/email-spam-detection-/assets/142707684/5b1357b2-0316-4463-af58-303be27b1957)


![image](https://github.com/ShrishtiSingh26/email-spam-detection-/assets/142707684/45e77015-2cba-451d-8066-22f80c1992bf)

This project classifies email messages as spam or ham using machine learning model Logistic regression...

Logistic regression is used for binary classification where we use sigmoid function, that takes input as independent variables and produces a probability value between 0 and 1.

![1_klFuUpBGVAjTfpTak2HhUA](https://github.com/ShrishtiSingh26/email-spam-detection-/assets/142707684/3a995b65-2a50-4b74-a9d6-68eda1661a29)

*The dataset used id mail_data.csv* 
Languages and Tools Required:
   - Programming Language: Python
   - Libraries: scikit-learn, Pandas, NumPy
   - Development Environment: any Python IDE (e.g., PyCharm, VSCode)


The model gives upto *96.70%* accuracy over training data and *96.59%* accuracy score for testing data.

Spam email detection is a crucial task in email security, as it helps users avoid unsolicited and potentially harmful emails. In this project, we developed a machine learning model to automatically classify emails as spam or legitimate (ham). By leveraging a labelled dataset and employing preprocessing techniques like text cleaning and numerical representation, we trained and evaluated several machine learning algorithms. Despite the inherent challenges such as dataset quality and the presence of false positives/negatives, our model demonstrated promising performance. Through optimization and fine-tuning, we were able to achieve better results, albeit with some limitations. Overall, this project underscores the significance of machine learning in combating spam emails and highlights the continuous effort required to enhance email security and user experience.
