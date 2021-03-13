# Install flask, sklearn, pandas, pickle
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import re
import string
import stopwordsiso as stopwords


app = Flask(__name__)

#Loading some model
categor = pd.read_csv('category.csv')
nb = pickle.load(open("random_forest_classi.pkl","rb"))
cv = pickle.load(open("cv_content.pkl","rb"))
cv_head = pickle.load(open("cv_head.pkl","rb"))
col_transform = pickle.load(open("one_hot.pkl","rb"))


# Stop words
stop_words = stopwords.stopwords("bn")



#  #####################  Function section  ########################

# NLP Preprocess function

# Apply a first round of text cleaning techniques
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# Pepocessing function for headline
def cleaning_headline(data):
    
    data['headline'] = data['headline'].str.replace('[A-Za-z]','')
    data['headline'] = data['headline'].str.replace('[।,.॥]','')
    data['headline'] = data['headline'].str.replace('[\xa0]','')
    data['headline'] = data['headline'].str.split(' ')
    data['headline'] = data['headline'].apply(lambda x: [item for item in x if item not in stop_words])
    data['headline'] = data['headline'].apply(lambda x: ' '.join(word for word in x))
  
    return data



# Pepocessing function for  content
def cleaning_content(data):
    
    data['content'] = data['content'].str.replace('[A-Za-z]','')
    data['content'] = data['content'].str.replace('[।,.॥]','')
    data['content'] = data['content'].str.replace('[\xa0]','')
    data['content'] = data['content'].str.split(' ')
    data['content'] = data['content'].apply(lambda x: [item for item in x if item not in stop_words])
    data['content'] = data['content'].apply(lambda x: ' '.join(word for word in x))
  
    return data



# Removing Emojies in content
def demoji_cont(text):
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
                              "]+", flags=re.UNICODE)
    
    return(emoji_pattern.sub(r'', text))
 



# Removing Emojies in headline
def demoji_head(text):
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
                              "]+", flags=re.UNICODE)
    
    return(emoji_pattern.sub(r'', text))




# ################ Flask Section #################



@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        category = myDict['category']
        relation = myDict['relation']
        headline = myDict['headline']
        content = myDict['content']
        
        # print(category)
        # print(relation)
        # print(headline)
        # print(content)


        inputs = pd.DataFrame([[category,relation,headline,content]], columns=['category','relation','headline','content'])

        inputs['relation'] = inputs['relation'].str.lower()
        inputs['category'] = inputs['category'].str.lower()

        inputs['headline'] = inputs['headline'].apply(round1)
        inputs['headline'] = inputs['headline'].apply(round2)
        inputs['content'] = inputs['content'].apply(round1)
        inputs['content'] = inputs['content'].apply(round2)

        inputs = cleaning_headline(inputs)
        inputs = cleaning_content(inputs)

        inputs['content'] = inputs['content'].apply(lambda x:demoji_cont(x))
        inputs['headline'] = inputs['headline'].apply(lambda x:demoji_head(x))

        testing = inputs.drop(columns=['category','relation']).values

        test_body = cv.transform(testing[:,1]).todense()
        print(test_body.shape)
        test_head = cv_head.transform(testing[:,0]).todense()
        print(test_head.shape)


        test_final = np.hstack(( test_head, test_body))
        print(test_final.shape)

        test_dum = inputs.drop(columns=['headline','content'])
        test_dumi = col_transform.transform(test_dum)
        print(test_dumi.shape)

        lets_test = np.hstack(( test_dumi, test_final))
        print(lets_test.shape)
        
        output = nb.predict(lets_test)[0]
        output = int(output)
        print(output)

        return render_template("show.html", prediction = output)

   
    categories = sorted(categor['category'].unique())
    relations = ['Related', 'Unrelated']	
    return render_template("index.html", locations = categories, relation = relations )

if __name__ == "__main__":
    app.run(debug=True)

        

   


