import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
paris1 = pickle.load(open('paris1.pkl','rb'))
paris2 = pickle.load(open('paris2.pkl','rb'))
paris3 = pickle.load(open('paris3.pkl','rb'))
paris4 = pickle.load(open('paris4.pkl','rb'))
paris10 = pickle.load(open('paris10.pkl','rb'))

@app.route('/')
def home():
    loyer = [0,0,0,0,0]
    return render_template('index.html', Loyer=loyer)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    para = [int(x) for x in request.form.values()]
    param=[np.array(para)]
    prediction1 = paris1.predict(param)
    prediction2 = paris2.predict(param)
    prediction3 = paris3.predict(param)
    prediction4 = paris4.predict(param)
    prediction10 = paris10.predict(param)
    
    output1 = np.round(prediction1[0][0], 2)
    output2 = np.round(prediction2[0][0], 2)
    output3 = np.round(prediction3[0][0], 2)
    output4 = np.round(prediction4[0][0], 2)
    output10 = np.round(prediction10[0][0], 2)
    
    loyer = [output1, output2, output3, output4, output10]

    return render_template('index.html', Loyer=loyer)

if __name__ == "__main__":
    app.run(debug=True)
