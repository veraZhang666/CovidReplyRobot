from flask import Flask, render_template, request, jsonify
import torch
import spacy
import pandas as pd


model=torch.nn.Linear(768,6) #只使用一层线性分类器
model = model.float()
model.load_state_dict(torch.load('classify_model1.pth'))


# =====================prepare text sentences=================
intent2label = {'greetings': 0, 'information': 1, 'prevention': 2,
                'symptoms': 3, 'travel': 4, 'vaccine': 5}
def label_to_intent(label):
    dic = {0:'greetings',1:'information',2:'prevention',
                    3:'symptoms', 4:'travel',5:'vaccine'}
    print(dic[label])
    return dic[label]

from sentence_transformers  import SentenceTransformer

def get_vec_from_Sentence(question):
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)
    sent_Vectors = model.encode(question)
    X = torch.from_numpy(sent_Vectors).unsqueeze(0)
    return X
respon_df = pd.read_csv('response.csv')

response_dic = {}
for i in range(7):
    response_dic[respon_df.iloc[i,0]] = respon_df.iloc[i,1]

def respond_ml(question):
    X = get_vec_from_Sentence(question)
    X = X.to('cpu')
    model.to('cpu')
    Y_pred = model(X)

    y_softmax = torch.softmax(Y_pred,dim=1)
    pred_label = torch.argmax(y_softmax,dim=1)

    intent = label_to_intent(pred_label.item())
    response = response_dic[intent]
    print('intention:',intent)
    return response

#============================Flask===========================
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat1.html')

@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText']
    print(message)
    while True:
        if message == "quit":
            exit()
        else:
            bot_response = respond_ml(message)
            print(bot_response)
            return jsonify({'status':'OK','answer':bot_response})
app.run(port=5001)

