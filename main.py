from flask import Flask, render_template,request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods = ["GET","POST"])

def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bpain = int(myDict['bpain'])
        cough = int(myDict['cough'])
        breathing = int(myDict['breathing'])
        chestPain = int(myDict['chestPain'])
        history = int(myDict['history'])

        # code for inference
        inputFeatures = [fever,age,bpain,cough,breathing,chestPain,history]
        infProba = clf.predict_proba([inputFeatures])[0][1]
        print(infProba)
    return render_template('index.html')

    #return'hello'+ str(infProb)
if __name__ == '__main__':
    app.run(debug=True)

