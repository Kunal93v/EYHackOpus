
# A very simple Flask Hello World app for you to get started with...

import os
#import magic
import urllib.request
import flask
from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
from flask import flash,request,send_file
from flask import Flask, render_template, session, redirect


app = Flask(__name__, template_folder='templates')


from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename



UPLOAD_FOLDER = r'/home/kunal93v/Upload/'


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['csv', 'xlsx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/match_demand', methods = ["GET","POST"])
def match_demand():

    # Import required libraries
    import pandas as pd
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re
    from gensim import utils
    from gensim.models.doc2vec import LabeledSentence
    from gensim.models.doc2vec import TaggedDocument
    from gensim.models import Doc2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import accuracy_score
    from nltk import word_tokenize
    import spacy
    import en_core_web_sm
    import glob

    path  = os.getcwd()
    files = os.listdir(UPLOAD_FOLDER)

    files = [f for f in files if f.split(".")[1] in ['xls', 'xlsx']]
    Demand = pd.read_excel(UPLOAD_FOLDER + files[0] )

    nlp = en_core_web_sm.load()

    w_T = float(request.args['w_T'])
    w_F = float(request.args['w_F'])
    w_P = float(request.args['w_P'])
    w_L = float(request.args['w_L'])
    w_Exp = float(request.args['w_Exp'])
    w_Rnk = float(request.args['w_Rnk'])
    w_Bnch = float(request.args['w_Bnch'])





    xls = pd.ExcelFile('/home/kunal93v/PS1 - ES Hackathon_SampleData_AI In Capacity Management8421018.xlsx')
    Skill_Tree = pd.read_excel(xls, 'Skill_Tree')

    #Supply = pd.read_excel(xls, 'Supply')
    Supply = pd.read_excel('/home/kunal93v/simulated_data.xlsx')
    Supply = Supply.fillna('NA')
    Supply[['Sub Unit 1', 'Sub Unit 2', 'Sub Unit 3',
       'Skill']] = Supply[['Sub Unit 1', 'Sub Unit 2', 'Sub Unit 3',
       'Skill']].fillna('NA')


    Demand = Demand.drop(index = 3, axis = 0)
    Demand =  Demand.fillna('NA')


    Supply['Loc_Score'] = 0
    Supply['SL_Score'] = 0
    Supply['sub_SL_Score'] = 0
    Supply['SMU_Score'] = 0
    Supply['Min_Exp_Score'] = 0
    Supply['Rank_flag'] = 0


    Supply['Tech_Score'], Supply['Func_Score'], Supply['Proc_Score'], Supply['Cum_Score'],Supply['Requestor'], Supply['Overall_Skills'] = None, None, None, None, None, None
    Supply['Years of experience norm'], Supply['Bench Ageing (weeks) norm'] = None, None

    Demand['Best Bet'], Demand['Best fit'], Demand['Stretched fit'] =None, None, None
    Supply['Years of experience norm'] = (Supply['Years of experience'] - Supply['Years of experience'].min())/(Supply['Years of experience'].max() - Supply['Years of experience'].min())
    Supply['Bench Ageing (weeks) norm'] = (Supply['Bench Ageing (weeks)'] - Supply['Bench Ageing (weeks)'].min())/(Supply['Bench Ageing (weeks)'].max() - Supply['Bench Ageing (weeks)'].min())


    out_df = pd.DataFrame()
    for r in Demand.Requestor:
        TD = Demand[Demand.Requestor == r][['Technical Skill 1', 'Technical Skill 2', 'Technical Skill 3', 'Job Title']].values.tolist()[0]
        FD = Demand[Demand.Requestor == r][['Functional Skill 1','Functional Skill 2','Functional Skill 3', 'Job Title']].values.tolist()[0]
        PD = Demand[Demand.Requestor == r][['Process Skill 1','Process Skill 2','Process Skill 3', 'Job Title']].values.tolist()[0]


        TDt = ",".join([x for x in TD if x not in ['NA']])
        FDt = ",".join([x for x in FD if x not in ['NA']])
        PDt = ",".join([x for x in PD if x not in ['NA']])

        for E in list(set(Supply['Name/ID'])):
            SS = Supply[Supply['Name/ID'] == E][['Sub Unit 1', 'Sub Unit 2','Sub Unit 3','Skill']].values
            SS = list(set( [item for sublist in SS for item in sublist]))

            SSt = ",".join([x for x in SS if x not in ['NA']])


            Supply.loc[Supply['Name/ID'] == E, ['Tech_Score', 'Func_Score', 'Proc_Score'] ] = nlp(TDt).similarity(nlp(SSt)), nlp(FDt).similarity(nlp(SSt)), nlp(PDt).similarity(nlp(SSt))
            Supply['Cum_Skills_Score'] =(w_T*Supply['Tech_Score'] + w_F*Supply['Func_Score'] + w_P*Supply['Proc_Score'] )/(w_T + w_F + w_P)

            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E].City.values[0] ==  Demand[Demand.Requestor == r]['Location '].values[0]), 'Loc_Score'] = 1
            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E]['Service Line'].values[0] ==  Demand[Demand.Requestor == r]['Requestor Service Line'].values[0]), 'SL_Score'] = 1
            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E]['Sub Service Line'].values[0] ==  Demand[Demand.Requestor == r]['Requestor Sub ServiceLine'].values[0]), 'sub_SL_Score'] = 1
            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E]['SMU'].values[0] ==  Demand[Demand.Requestor == r]['Requestor SMU'].values[0]), 'SMU_Score'] = 1
            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E]['Years of experience'].values[0] >=  float(Demand[Demand.Requestor == r]['Min Experience'].values[0])), 'Min_Exp_Score'] = 1
            Supply.loc[(Supply['Name/ID'] == E) & (Supply[Supply['Name/ID'] == E]['Rank'].values[0] ==  Demand[Demand.Requestor == r]['Rank'].values[0]), 'Rank_flag'] = 1
            Supply.loc[Supply['Name/ID'] == E , 'Overall_Skills'] = SSt

            Supply['Cum_Score'] = (w_T*Supply['Tech_Score'] + w_F*Supply['Func_Score'] + w_P*Supply['Proc_Score'] + w_L*Supply['Loc_Score'] +\
                w_Exp*Supply['Min_Exp_Score']*Supply['Years of experience norm'] + w_Rnk*Supply['Rank_flag'] + w_Bnch*Supply['Bench Ageing (weeks) norm'] +\
            10*Supply['SL_Score'] + 10*Supply['sub_SL_Score'] + 10*Supply['SMU_Score'] + 10*Supply['Skill Level'])/(w_T + w_F + w_P + w_L +\
                                                                                                                   w_Exp + w_Rnk + w_Bnch + 40)
            Supply['Requestor'] = r
            #Supply = Supply.sort_values(by = ['Loc_Score','SMU_Score', 'sub_SL_Score', 'SL_Score','Cum_Score'], ascending = [False, False, False, False, False])
        out_df = out_df.append(Supply)




    out_df['Cum_Score'] = out_df['Cum_Score'].astype('float')
    out_df = out_df[['Requestor','Name/ID', 'Overall_Skills', 'Skill Level', 'Years of experience', 'Rank', 'Service Line',
           'Sub Service Line', 'SMU',  'City', 'Bench Ageing (weeks)',
           'Loc_Score', 'SL_Score', 'sub_SL_Score', 'SMU_Score',
           'Tech_Score', 'Func_Score', 'Proc_Score',
            'Cum_Skills_Score','Cum_Score']] #'Sub Unit 1', 'Sub Unit 2','Sub Unit 3','Skill',

    out_df = out_df[out_df['Cum_Score'] == out_df.groupby(['Requestor','Name/ID'])['Cum_Score'].transform('max')]
    out_df = out_df.drop_duplicates()
    out_df = out_df.drop_duplicates(subset=['Name/ID', 'Cum_Score'], keep='first')

    M = out_df.Cum_Score.max()

    def fitment_class(x, m = M):
        if x/m >= 0.85:
            return 'Best Fit'
        elif (x/m < 0.85) & (x/m > 0.7):
            return 'Stretched Fit'

        elif (x/m < 0.7) & (x/m >0.6):
            return 'Best Bet'

        else:
            return 'No Fit'

    out_df['fitment_class'] = out_df.Cum_Score.apply(fitment_class)
    out_df['fitment_percentage'] = 100*out_df.Cum_Score/M
    out_df = out_df.sort_values(by = ['Requestor','fitment_percentage'], ascending = [True,  False])

    out_df.to_csv('/home/kunal93v/mysite/out.csv' , index = None, header=True)
    return render_template('simple.html',   tables=[out_df.to_html(classes='data', header="true")])



@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

@app.route('/download')
def download_file():
    p = 'out.csv'
    return send_file(p, as_attachment = True , cache_timeout=0)
