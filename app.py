import os
import MySQLdb
import smtplib
import random
import string
import np
from datetime import datetime
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash, send_file
from database import db_connect,inc_reg,ins_loginact
# Import Libraries
import numpy as num
import pandas as pd
import matplotlib.pyplot as graph
import seaborn as sb
import sys
import io
import json 
from sendmail import sendmail
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# def db_connect():
#     _conn = MySQLdb.connect(host="localhost", user="root",
#                             passwd="root", db="assigndb")
#     c = _conn.cursor()

#     return c, _conn
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
 


selected_features = ['chains', 'glasses', 'pants', 'rings','shoes','T Shirt','watches']

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def FUN_root():
    return render_template("index.html")
    
@app.route("/1.html")
def admin():
    return render_template("1.html")

@app.route("/men.html")
def men():
    return render_template("men.html")

@app.route("/women.html")
def women():
    return render_template("women.html")

@app.route("/kids.html")
def kids():
    return render_template("kids.html")

@app.route("/kidsg.html")
def Gkids():
    return render_template("kidsg.html")

@app.route("/user.html")
def ins():
    return render_template("user.html")


@app.route("/increg.html")
def increg():
    return render_template("increg.html")





@app.route("/ihome.html")
def ihome():
    return render_template("ihome.html")


@app.route("/p.html")
def p():
    return render_template("p.html")



@app.route("/index")
def index():
    return render_template("index.html") 






# -------------------------------Registration-----------------------------------------------------------------    




@app.route("/inceregact", methods = ['GET','POST'])
def inceregact():
   if request.method == 'POST':    
      
      status = inc_reg(request.form['username'],request.form['password'],request.form['email'],request.form['mobile'])
      
      if status == 1:
       return render_template("user.html",m1="sucess")
      else:
       return render_template("increg.html",m1="failed")


@app.route("/menact", methods = ['GET','POST'])
def menact():
   if request.method == 'POST':    
      
      chest = int(request.form['chest'])
      shoulders = int(request.form['shoulders'])
      dlength = float(request.form['dlength'])
      dsize = int(request.form['dsize'])
      ashoulder = float(request.form['ashoulder'])
      slength = int(request.form['slength'])
      dcolour = request.form['dcolour']
      Category = request.form['Category']

      print(Category)
      Features_labels = pd.read_csv("fd.csv")
      
      DatasetFeatureNames = ['chest', 'sholders', 'length', 'size', 'across sholder', 'sleeve length', 'colour','Category','Label']  
      featuredata_X = Features_labels.iloc[:, :-1].values
      Labeldata_y= Features_labels.iloc[:, 8].values
      from sklearn.preprocessing import LabelEncoder
      trainx=LabelEncoder()      
      featuredata_X[:, 6] = trainx.fit_transform(featuredata_X[:, 6])  # Encode 'colour'
      featuredata_X[:, -1] = trainx.fit_transform(featuredata_X[:, -1])  # Encode 'Category'
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(featuredata_X, Labeldata_y,test_size=0.2, random_state=69)  
      from sklearn.linear_model import LogisticRegression
      LRmodel = LogisticRegression(solver='lbfgs')
      LRmodel.fit(X_train, y_train)
      color=0
      if dcolour == 'white':
         color=0
      if dcolour == 'Black':
         color=1
      if dcolour == 'medium':
         color=2
      check = np.array([chest, shoulders, dlength, dsize, ashoulder, slength, color, trainx.transform([Category])[0]]).reshape(1, -1)

        # Make prediction using the trained model
      pred = LRmodel.predict(check)[0]  # Get the first prediction

        # Determine output based on prediction
      if pred == 1:
            return render_template("1.html", m1="success", da=Category+'mone')
      elif pred == 2:
            return render_template("2.html", m1="failed", da=Category+'mtwo')
      else:
            return render_template("3.html", m1="failed", da=Category+'mthree')

@app.route("/mewomenactnact", methods = ['GET','POST'])
def womenact():
   if request.method == 'POST':    
      
      bust = int(request.form['bust'])
      waist = int(request.form['waist'])
      hip = int(request.form['hip'])
      shoulders = float(request.form['shoulders'])
      slevers = float(request.form['slevers'])
      dsize =  request.form['dsize']
      dcolour = request.form['dcolor']
      wDatasetFeatureNames = ['Bust', 'waist', 'hip', 'sholders', 'slevers', 'dsize','colour', 'Label']
      wFeatures_labels=pd.read_csv("fd1.csv")
       
      wfeaturedata_X = wFeatures_labels.iloc[:, :-1].values
      wLabeldata_y= wFeatures_labels.iloc[:, 7].values
      from sklearn.preprocessing import LabelEncoder
      wtrainx=LabelEncoder()
      wfeaturedata_X[:,5]=wtrainx.fit_transform(wfeaturedata_X[:,5])
      wfeaturedata_X[:,-1]=wtrainx.fit_transform(wfeaturedata_X[:,-1])      
      from sklearn.model_selection import train_test_split
      X_wtrain, X_wtest, y_wtrain, y_wtest = train_test_split(wfeaturedata_X, wLabeldata_y,test_size=0.2, random_state=69)
      from sklearn.linear_model import LogisticRegression
      wLRmodel = LogisticRegression(solver='lbfgs')
      wLRmodel.fit(X_wtrain, y_wtrain)
      color=0
      if dcolour == 'white':
         color=0
         var=dsize+'wone'
      if dcolour == 'Black':
         color=1
         var=dsize+'wtwo'
      if dcolour == 'medium':
         color=2
         var=dsize+'wthree'
      import numpy as np
      print(var)       
      dsize_encoded = wtrainx.fit_transform([dsize])[0]
      check1 = np.array([bust,waist, hip, shoulders, slevers, dsize_encoded,color]).reshape(1, -1)
      pred = wLRmodel.predict(check1)[0] 
      
      if pred == 1:
       return render_template("w1.html",m1="sucess",da=var) 
      elif pred == 2 :
       return render_template("w2.html",m1="failed",da=var)
      else:
        return render_template("w3.html",m1="failed",da=var)


@app.route("/kids1act", methods = ['GET','POST'])
def kids1act():
   if request.method == 'POST':    
      
      chest=int(request.form['chest'])
      shoulders = int(request.form['shoulders'])
      dlength = float(request.form['dlength'])
      dsize = int(request.form['dsize'])
      ashoulder = float(request.form['ashoulder'])
      slength = int(request.form['slength'])
      dcolour = request.form['dcolour']
      Category = request.form['Category']

      print(Category)
      Features_labels = pd.read_csv("fd4.csv")
      
      DatasetFeatureNames = ['chest','sholders','length','size','across sholder','sleeve length','colour','Category']
      featuredata_X = Features_labels.iloc[:, :-1].values
      Labeldata_y= Features_labels.iloc[:, 8].values
      from sklearn.preprocessing import LabelEncoder
      trainx=LabelEncoder()      
      featuredata_X[:, 6] = trainx.fit_transform(featuredata_X[:, 6])  # Encode 'colour'
      featuredata_X[:, -1] = trainx.fit_transform(featuredata_X[:, -1])  # Encode 'Category'
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(featuredata_X, Labeldata_y,test_size=0.2, random_state=69)  
      from sklearn.linear_model import LogisticRegression
      LRmodel = LogisticRegression(solver='lbfgs')
      LRmodel.fit(X_train, y_train)
      color=0
      if dcolour == 'white':
         color=0
         var=Category+'bone'
      if dcolour == 'Black':
         color=1
         var=Category+'btwo'
      if dcolour == 'medium':
         color=2
         var=Category+'bthree'
         
      check2 = np.array([chest,shoulders, dlength, dsize, ashoulder, slength, color, trainx.transform([Category])[0]]).reshape(1, -1)

        # Make prediction using the trained model
      pred = LRmodel.predict(check2)[0]  # Get the first prediction

        # Determine output based on prediction
      if pred == 1:
            return render_template("k1.html", m1="success", da=var)
      elif pred == 2:
            return render_template("k2.html", m1="failed", da=var)
      else:
            return render_template("k3.html", m1="failed", da=var)
        
        
        
@app.route("/kidsgact", methods = ['GET','POST'])
def kidsgact():
    if request.method == 'POST':    
      
      chest=int(request.form['chest'])
      shoulders = int(request.form['shoulders'])
      dlength = float(request.form['dlength'])
      dsize = int(request.form['dsize'])
      ashoulder = float(request.form['ashoulder'])
      slength = int(request.form['slength'])
      dcolour = request.form['dcolour']
      Category = request.form['Category']

      print(Category)
      Features_labels = pd.read_csv("fd4.csv")
      
      DatasetFeatureNames = ['chest','sholders','length','size','across sholder','sleeve length','colour','Category']
      featuredata_X = Features_labels.iloc[:, :-1].values
      Labeldata_y= Features_labels.iloc[:, 8].values
      from sklearn.preprocessing import LabelEncoder
      trainx=LabelEncoder()      
      featuredata_X[:, 6] = trainx.fit_transform(featuredata_X[:, 6])  # Encode 'colour'
      featuredata_X[:, -1] = trainx.fit_transform(featuredata_X[:, -1])  # Encode 'Category'
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(featuredata_X, Labeldata_y,test_size=0.2, random_state=69)  
      from sklearn.linear_model import LogisticRegression
      LRmodel = LogisticRegression(solver='lbfgs')
      LRmodel.fit(X_train, y_train)
      color=0
      if dcolour == 'white':
         color=0
         var=Category+'bone'
      if dcolour == 'Black':
         color=1
         var=Category+'btwo'
      if dcolour == 'medium':
         color=2
         var=Category+'bthree'
         
      check2 = np.array([chest,shoulders, dlength, dsize, ashoulder, slength, color, trainx.transform([Category])[0]]).reshape(1, -1)

        # Make prediction using the trained model
      pred = LRmodel.predict(check2)[0]  # Get the first prediction

        # Determine output based on prediction
      if pred == 1:
            return render_template("gk1.html", m1="success", da=var)
      elif pred == 2:
            return render_template("gk2.html", m1="failed", da=var)
      else:
            return render_template("gk3.html", m1="failed", da=var)
         
# #-------------------------------ADD_END---------------------------------------------------------------------------
# # -------------------------------Loginact-----------------------------------------------------------------







@app.route("/inslogin", methods=['GET', 'POST'])       
def inslogin():
    if request.method == 'POST':
        status = ins_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']
            return render_template("ihome.html", m1="sucess")
        else:
            return render_template("user.html", m1="Login Failed")
        



# # -------------------------------Loginact End----------------------------------------------------------------



@app.route('/pr', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['image']
        print("ddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        print(image_file)
        filename = secure_filename(image_file.filename)
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Read the image using Pillow
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        classes = {1:'chains', 2:'glasses', 3:'pants', 4:'rings',5:'shoes',0:'T Shirt',6:'watches'}
     
        # load the model we saved
        model = load_model('fan.h5')
      
        result = np.argmax(model.predict(image))
        print(result)       
        prediction1 = classes[result]
        print(prediction1)
        image_folder = os.path.join(app.config['UPLOAD_FOLDER'],prediction1)
        images_info = []
        for file in os.listdir(image_folder):
            if file.endswith(('jpg', 'jpeg', 'png', 'gif')):
                images_info.append({'filename': file, 'path': os.path.join(image_folder, file)})
        # Render the HTML template with the prediction result and image
        return render_template('predictionoutput.html', prediction={'p':prediction1,'image':os.path.join(app.config['UPLOAD_FOLDER'], filename),'images_info':images_info})



   
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
