import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens


urls_data = pd.read_csv('data/urldata.csv', error_bad_lines=False)
type(urls_data)
urls_data = pd.DataFrame(urls_data)
urls_data = urls_data[pd.notnull(urls_data['url'])]
totalurlslist = urls_data
totalurlslist = np.array(totalurlslist)
random.shuffle(totalurlslist)
random.shuffle(totalurlslist)
random.shuffle(totalurlslist)
myUrls = [item[0]for item in totalurlslist] #urls
y = [item[1]for item in totalurlslist] #labels

# Pre-processes URLs by removing protocol portion of each URL
processedUrls = [x.replace('https://', '').replace('http://', '') for x in myUrls]

vectorizer = TfidfVectorizer( tokenizer=makeTokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
X = vectorizer.fit_transform(processedUrls)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=64)

# hyperparameter tuning
# grid search parameters
param_grid_lr = {
    'max_iter': [ 1000 ],
    'solver': ['newton-cg', 'lbfgs', 'liblinear',],
    'class_weight': ['balanced'],
    'C' : [10, 0.01],
}
# hyperparameter action and results
logModel_grid = GridSearchCV(estimator=LogisticRegression(random_state=1234), param_grid=param_grid_lr, verbose=1, cv=10, n_jobs=-1)
logModel_grid.fit(X_train, y_train)
print(logModel_grid.best_estimator_)


clf = LogisticRegression(max_iter=1000, random_state=1234, class_weight='balanced', C=10)
clf.fit(X_train, y_train)
accuracy = 100*clf.score(X_test, y_test)
accuracy = str(round(accuracy, 2))
print('Test accuracy:', accuracy, "%")
BAD_len = urls_data[urls_data["label"] == "bad"].shape[0]
GOOD_len = urls_data[urls_data["label"] == "good"].shape[0]
print('Malicious URL count:', BAD_len)
print('Good URL count:', GOOD_len)


testURL = input('Please enter a URL to test it')
X_predict = [testURL]
X_predict = vectorizer.transform(X_predict)
y_predict = clf.predict(X_predict)
print(y_predict)


plt.bar(10,BAD_len,3, label= "BAD URL")
plt.bar(15,GOOD_len,3, label= "GOOD URL")
plt.legend()
plt.ylabel("Number of examples")
plt.title("Proportion of examples")
plt.show()
