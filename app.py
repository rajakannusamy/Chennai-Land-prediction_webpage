from copy import error
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['LotArea']
    val2 = request.form['OverallQual']
    val3 = request.form['OverallCond']
    val4 = request.form['YearBuilt']
    
    try:
         arr = np.array([val1, val2, val3, val4])
         arr = arr.astype(np.float64)
         pred = model.predict([arr])
    except:
            pred=0
    return render_template('index.html', data=int(pred))
#Handling error 404 and displaying relevant web page
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'),404
 
#Handling error 500 and displaying relevant web page
@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'),500

if __name__ == '__main__':
    app.run()
