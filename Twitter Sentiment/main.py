from flask import Flask,render_template,request
import pickle
#Importing NLP libraries 
import re  #Regular Expression
from bs4 import BeautifulSoup
import nltk   
from nltk.corpus import stopwords

# donloading
nltk.download('punkt')
nltk.download('stopwords')
# Reading english stopwords
stop_words = set(stopwords.words('english'))


#Preprocess function
def preprocess_text(data):
    sentence = []

    for review in data:
        striped_review = review.strip()
        soup_review = BeautifulSoup(striped_review).get_text()
        review_text = re.sub('[^a-zA-Z]', ' ', soup_review)
        words = review_text.lower().split()

        clean_words = [word for word in words if not word in stop_words]  #Avoidong stop words

        join_str = ''

        for w in clean_words:
            join_str = join_str + ' ' + w
        
        sentence.append(join_str.strip())

    return sentence




app = Flask(__name__)


model = pickle.load(open("nb_multi.pkl","rb"))

@app.route('/', methods=["GET","POST"])
def main_function():
    if request.method == "POST":
        text = request.form
        # print(text)
        twieet = text['twieet']
        print(twieet)
        texts = [twieet]
        prepose = preprocess_text(texts)
        output = model.predict(prepose)


        return render_template("show.html", prediction = output)
    
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)