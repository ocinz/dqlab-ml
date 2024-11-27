import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from PIL import Image
import time
pd.set_option('future.no_silent_downcasting', True)

url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names, skiprows=[0])

@st.cache_resource
def import_model(model_name):
    with open(model_name, 'rb') as file:  
            return pickle.load(file)
    
            
# utils functions
def outliers(data, data_out):
        for each_feature in data_out.columns:
            feature_data = data_out[each_feature]
            Q1 = np.percentile(feature_data, 25.) 
            Q3 = np.percentile(feature_data, 75.) 
            IQR = Q3-Q1 
            outlier_step = IQR * 1.5 
            outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
            data.drop(outliers, inplace = True, errors = 'ignore')

# page functions
def overview ():
    st.write("""
             ### Dataset Overview
             """)
    """
    ##### Information
    This database contains 76 attributes, but all published experiments refer to using a subset of 14
    of them.  
    """
    """
    ##### Data Variables & Types
    """
    feature = pd.DataFrame({
        "Varibale Name": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    , "Role": ["Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature","Feature", "Target"]
    , "Type": ["Integer", "Categorical", "Categorical", "Integer", "Integer","Categorical","Categorical","Integer","Categorical","Integer","Categorical","Integer", "Categorical","Integer"]
    , "Missing Values": ["no","no","no","no","no","no","no","no","no","no","no","yes", "yes", "no"]
    , "Description": ["Age in years", "Gender: (1: Male, 0: Female)", "chest pain type: (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic", "resting blood pressure (in mm Hg)", "serum cholestoral in mg/dl", "fasting blood sugar>120mg/dl:(1:true, 0:false)", "resting electrocardiographic results: (0:normal,1:ST-T wave abnormality,2:left ventricular hypertrophy", "maximum heart rate achieved", "exercise induced angina: (1:yes, 0:no)", "ST depression induced by exercise relative to rest", "the slope of the peak exercise ST segment: 1: upsloping, 2: flat, 3:downsloping", "number of major vessels (0-3) colored by flourosopy", "3:normal,6:fixed defect, 7:reversable defect", "target"]
    })
    st.dataframe(feature, hide_index=True, width=2000)
    
    """
    ##### Data Shape & Sneakpeek
    """
    "Rows and cols:" 
    data.shape
    "10 First Data:"
    st.dataframe(data.head(10), hide_index=True)
    
def pre_processing():
    st.write("""
             ### Data Preprocessing and Exploratory Data Analysis
             """)
    
    st.write("""
             ##### Data Statistic
             """)
    st.dataframe(data.describe())
    st.write("""
             This statistical data appears to be normal, but we will explore further to gain deeper insights.
             """)
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
                ##### Data Types
                """)
        st.dataframe(pd.DataFrame(data.dtypes), width=500)
        st.write("All data types appear to be numerical, including categorical data that has been converted to numbers. We will attempt to convert them to their appropriate types.")
    with col2:
        st.write("""
                ##### Unique Values
                """)
        st.dataframe(pd.DataFrame(data.nunique()), width=500)
        st.write("Some categorical columns exceed the unique value limits specified by UCI, indicating a potential human error.")
    
    # Mengubah data type
    st.write("""
             ##### Data Labeling
             """)
    lst=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca', 'target']
    data[lst] = data[lst].astype(object)
    # Pelabelan data categorical
    data['sex'] = data['sex'].replace({1: 'Male',
                                    0: 'Female'})
    data['cp'] = data['cp'].replace({0: 'typical angina',
                                    1: 'atypical angina',
                                    2: 'non-anginal pain',
                                    3: 'asymtomatic'})
    data['fbs'] = data['fbs'].replace({0: 'No',
                                    1: 'Yes'})
    data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy',
                                            1:'normal',
                                            2: 'ST-T Wave abnormal'})
    data['exang'] = data['exang'].replace({0: 'No',
                                        1: 'Yes'})
    data['slope'] = data['slope'].replace({0: 'downsloping',
                                        1: 'flat',
                                        2: 'upsloping'})
    data['thal'] = data['thal'].replace({1: 'normal',
                                        2: 'fixed defect',
                                        3: 'reversable defect'})
    data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                    1: 'Number of major vessels: 1',
                                    2: 'Number of major vessels: 2',
                                    3: 'Number of major vessels: 3'})
    data['target'] = data['target'].replace({0: 'No disease',
                                            1: 'Disease'})
    st.dataframe(pd.DataFrame(data.head(10)))
    st.write("Converting the categorical data back to its original categorical format instead of using numbers to represent categories.")
    
    st.write("""
             ##### Converting Invalid Data to `NaN`
             """)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame(data.ca.value_counts()), width=500)
    with col2:
        st.dataframe(pd.DataFrame(data.thal.value_counts()), width=500)
    st.write("Strange data has been identified in the `ca` and `thal` columns and will be converted to `NaN`. These `NaN` values will later be handled by filling them using central tendency measures such as mode, median, or mean.")
    
    data.loc[data['ca']==4, 'ca'] = np.nan
    data.loc[data['thal']==0, 'thal'] = np.nan
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
                ##### Missing Values
                """)
        st.dataframe(pd.DataFrame(data.isnull().sum()), width=500)
    
    modus_ca = data['ca'].mode()[0]
    data['ca'] = data['ca'].fillna(modus_ca)
    modus_thal = data['thal'].mode()[0]
    data['thal'] = data['thal'].fillna(modus_thal)
    
    with col2:
        st.write("""
                ##### Filling Missing Values
                """)
        st.dataframe(pd.DataFrame(data.isnull().sum()), width=500)
    st.write("`NaN` values have been filled with the mode, as it is less sensitive to `outliers`.")
        
    # duplicated data
    st.write("""
                ##### Handling Duplicate Data
                """)
    st.dataframe(pd.DataFrame(data[data.duplicated()]))
    data.drop_duplicates(inplace=True, keep='last')
    st.write("We will drop the duplicates and retain the last occurrence of each duplicate.")
    
    
    # outliers
    st.write("""
                ##### Handling `Outliers`
                """)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    data.plot(kind='box', subplots=True, ax=axes, color='k', layout=(2,5))
    st.pyplot(fig)
    continous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']  
    outliers(data, data[continous_features])
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    data.plot(kind='box', subplots=True, ax=axes, color='k', layout=(2,5))
    st.pyplot(fig)
    st.write("Small data points have been identified as `outliers` and will be removed from the dataset.")
    
    # distribusi data
    st.write("""
                ##### Distribution of Data
                """)
    numerical_col = data.select_dtypes(exclude=['object'])
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    fig.tight_layout(pad=4.0)
    axes = axes.flatten()
    for i, col in enumerate(numerical_col.columns[:5]):
        sns.histplot(numerical_col[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    for i in range(5, len(axes)):
        fig.delaxes(axes[i])
    st.pyplot(fig)
    st.write("The histogram above shows that most of the data follows a normal distribution or is close to normal.")
    
    # corr matrix
    st.write("""
                ##### Correlation Heatmap
                """)
    data['sex'] = data['sex'].replace({'Male' : 1,'Female': 0})
    data['cp'] = data['cp'].replace({'typical angina' : 0, 'atypical angina' : 1, 'non-anginal pain' : 2, 'asymtomatic' : 3})
    data['fbs'] = data['fbs'].replace({'No' : 0, 'Yes' : 1})
    data['restecg'] = data['restecg'].replace({'probable or definite left ventricular hypertrophy':0,'normal':1,'ST-T Wave abnormal':2})
    data['exang'] = data['exang'].replace({'No':0,'Yes':1})
    data['slope'] = data['slope'].replace({'downsloping':0, 'flat':1,'upsloping':2})
    data['thal'] = data['thal'].replace({'normal':1, 'fixed defect':2,'reversable defect':3})
    data['ca'] = data['ca'].replace({'Number of major vessels: 0':0, 'Number of major vessels: 1':1,'Number of major vessels: 2':2, 'Number of major vessels: 3':3})
    data['target'] = data['target'].replace({'No disease':0,'Disease':1})
    
    cor = data.corr()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    sns.heatmap(cor, annot=True, linewidth=.5, cmap="magma", ax=axes)
    # plt.title('Korelasi Antar Variable', fontsize = 30)
    st.pyplot(fig)
    st.write("The heatmap above illustrates the correlations between features in the dataset. Darker or lighter colors represent the strength of the correlation, with values closer to +1 or -1 indicating a strong relationship. This visualization helps identify significant feature relationships for further analysis or feature selection in modeling.")
    
    # filter
    st.write("""
             ##### Choosing Feature Data for the Machine Learning Model
            """)
    cor_matrix = data.corr()
    st.dataframe(pd.DataFrame(cor_matrix['target'].sort_values()), width=200)
    
    """
    From the table above, we can select the features with strong correlations to the target as follows:\n
    `cp`, `thalach`, `slope`, `oldpeak`, `exang`, `ca`, `thal`, `sex`, `age`.
    """

def modelling():
    st.write("""
             ### Modelling
             """)
    
    st.write('''
        ##### Model Accuracy Before Tuning
        ''')
    "Modeling was conducted using several algorithms, and the resulting accuracy for each algorithm is as follows:"
    accuracy_score = pd.DataFrame({
        "Model Algorithm": ["Logistic Regression", "Decision Tree", "Random Forest", "MLP Classifier"],
        "Accuracy %": [85, 78, 87, 87]
        })
    st.dataframe(accuracy_score, hide_index=True, width=500)

    accuracy_score = pd.DataFrame({
        "Model Algorithm": ["Logistic Regression", "Decision Tree", "Random Forest", "MLP Classifier"],
        "Accuracy %": [87, 83, 88, 89]
        })
    
    st.write('''
        ##### Model Accuracy After Tuning Using GridSearch
        ''')
    st.dataframe(accuracy_score, hide_index=True, width=500)
    st.write('''
        Multi-layer Perceptron Classifier will be chosen as the prediction model because it achieved the highest accuracy of 89% after tuning. 
        ''')
    
def prediction():
    st.write("""
             ###### Prediction
             """)
    
    def user_input_features():
            st.sidebar.header('Prediction Input')
            chestpain_type = ["Angina CP", "Unstable CP", "Worst Unstable CP", "Unrelated CP"]
            cp = chestpain_type.index(st.sidebar.selectbox('Chest Pain', chestpain_type)) + 1
            thalach = st.sidebar.slider("Max. Heart rate", 71, 202, 80)
            slope = st.sidebar.slider("The inclination of the ST segment in the ECG", 0, 2, 1)
            oldpeak = st.sidebar.slider("How much the ST segment decreases", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise Induced Angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of Major Vessels", 0, 3, 1)
            thal = st.sidebar.slider("Thalium Test", 1, 3, 1)
            gender_op = ['Female', 'Male']
            sex = gender_op.index(st.sidebar.selectbox("Gender", gender_op))
            age = st.sidebar.slider("Age", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
        
    input_df = user_input_features()
    img = Image.open("heart-disease.jpg")
    st.image(img, width=300)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        loaded_model = import_model("selected_model.pkl")
        prediction = loaded_model.predict(df)        
        st.subheader('Prediction: ')
        with st.spinner('Wait for it...'):
            time.sleep(2)
            match prediction:
                case 0: st.success("Prediction result: No Heart Disease")
                case _: st.warning("Prediction result: Yes Heart Disease")
    
def about(): 
    st.write("""
             ### About
             """)
    st.write("""
            
            Name : Ahmad Hafid\n
            **Spring Boot & Next JS Developer | Cyber Security & Data Science Enthusiast**\n
            Domicile: Malang City, Indonesia \n
            ahmadhafid28632@gmail.com | [LinkedIn](https://www.linkedin.com/in/ahmad-hafid/) | [Github](https://github.com/ocinz) | [Scholar](https://scholar.google.com/citations?user=t73daCgAAAAJ&hl=id) 

            This page was created as part of the **[DQLab](https://dqlab.id) Machine Learning Bootcamp Capstone Project, Batch 14 - 2024**.
             """)

# metadata
st.set_page_config(
    page_title="Heart Disease - DQLab Capstone Project",
    menu_items={
        'About': "Created By [Ahmad Hafid](https://linkedin.com/in/ahmad-hafid)",
    },
    page_icon=":shark:"
)
st.title("Heart Disease - DQLab Capstone Project")
"""
Heart Disease Modelling by [Ahmad Hafid (Ocinz)](https://www.linkedin.com/in/ahmad-hafid/). Credit: Data provided by [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
"""

# sidebar
st.sidebar.text("DQLab Capstone Project - Heart Disease")
navigation = st.sidebar.selectbox(
    'Navigation',
    ('Overview', 'Data Pre-processing', 'Modelling', 'Prediction', 'About'))

match navigation:
    case 'Overview':
        overview()
    case 'Data Pre-processing':
        pre_processing()
    case 'Modelling':
        modelling()
    case 'Prediction':
        prediction()
    case 'About':
        about()
    case _ :
        overview()