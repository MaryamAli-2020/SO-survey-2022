
!pip uninstall pandas-profiling

!pip install pandas-profiling[notebook,htm1]

pip install country_converter --upgrade

import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
import matplotlib.pyplot as plt
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import country_converter as coco
import plotly.express as px
from plotly import offline as pyo
from plotly import graph_objs as go

df = pd.read_csv('/content/survey_results_public.csv')
df

df.info()

df_copy = df.copy()
selected_columns = ['Country','Age','Gender','EdLevel','YearsCode','YearsCodePro','LanguageHaveWorkedWith','LanguageWantToWorkWith','LearnCodeCoursesCert','ProfessionalTech','SOAccount','Employment','DevType','WorkExp',]

df = df[selected_columns]
df

df.shape[0]

index = 0
for i in df.isnull().sum(axis=1):
  if i >=13:
    df.drop(index, inplace=True)
  index+=1

print(df.duplicated().sum(), 'duplicated rows have been dropped')
df.drop_duplicates(inplace=True)
print(df.shape[0],'rows and', df.shape[1],'cols remaining')

df = df.reset_index()
df

df.drop(columns='index', inplace=True)

df.insert(8, 'KnowAndWantToWorkWith', None)

for i in df.index:
    one = str(df.loc[i,'LanguageHaveWorkedWith']).split(';')
    two = str(df.loc[i,'LanguageWantToWorkWith']).split(';')
    three = list(set(one).intersection(set(two)))
    if(three != ''):
      df.loc[i,'KnowAndWantToWorkWith'] = ";".join(three)

x = str(df['KnowAndWantToWorkWith'].to_list())
x

x = str(df['KnowAndWantToWorkWith'].to_list())

to_remov = {"'": "", ";": " ", ",": "", "[": "", "]":"",}

for char in to_remov.keys():
    x = x.replace(char, to_remov[char])

x = x.lower()
x = re.sub(r"\s+", " ", x)
x

getUnique = pd.DataFrame(x.split(' '))
getUnique[0].unique()

stp = ['nan']
wc = WordCloud(stopwords = stp,width = 1000, height = 500,collocations=False, min_font_size=10, regexp= r"\w[\w']*[+#/]*\w[\w']*").generate(x)

plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.title("WordCloud of Prog. Languages Developers Know and Would Like to Work With\n", fontdict={'fontsize': 29})
plt.show()

def makeWordCloud(colName):
  x = str(df[colName].to_list())

  to_remov = {"'": "", ";": " ", ",": "", "[": "", "]":"",}

  for char in to_remov.keys():
      x = x.replace(char, to_remov[char])

  x = x.lower()
  x = re.sub(r"\s+", " ", x)
  
  stp = ['nan' ]
  wc = WordCloud(stopwords = stp,width = 1000, height = 500,collocations=False, min_font_size=10, regexp= r"\w[\w']*[+#/]*\w[\w']*").generate(x)

  plt.figure(figsize=(20,10))
  plt.imshow(wc)
  plt.axis("off")
  plt.title("WordCloud of "+colName+"\n", fontdict={'fontsize': 29})
  plt.show()

makeWordCloud('LanguageHaveWorkedWith')

makeWordCloud('LanguageWantToWorkWith')

makeWordCloud('LearnCodeCoursesCert')

df['Country'].nunique()

cc = coco.CountryConverter()
df.insert(0, 'Standard_country_names',  cc.convert(names = df.Country, to = 'name_short'))

df['Standard_country_names'].nunique()

df[['Country', 'Standard_country_names']]

df.drop(columns='Country', inplace=True)

df['Standard_country_names']= df['Standard_country_names'].str.replace(' ','')
df['Standard_country_names']

makeWordCloud('Standard_country_names')

df.info()

df.YearsCodePro = df.YearsCodePro.replace('Less than 1 year','0')
df.YearsCodePro = df.YearsCodePro.replace('More than 50 years','51')
df.YearsCodePro =  df.YearsCodePro.fillna(0)
df.YearsCodePro = df.YearsCodePro.astype(int)

df.YearsCode = df.YearsCode.replace('Less than 1 year','0')
df.YearsCode = df.YearsCode.replace('More than 50 years','51')
df.YearsCode =  df.YearsCode.fillna(0)
df.YearsCode = df.YearsCode.astype(int)

df['Gender'] = df['Gender'].replace(['Or, in your own words:','Non-binary, genderqueer, or gender non-conforming','Man;Woman','Man;Woman;Non-binary, genderqueer, or gender non-conforming',
                      'Man;Or, in your own words:;Woman;Non-binary, genderqueer, or gender non-conforming','Man;Or, in your own words:;Woman',
                      'Or, in your own words:;Non-binary, genderqueer, or gender non-conforming'], 'Prefer not to say')

df['Gender'] = df['Gender'].replace(['Man;Non-binary, genderqueer, or gender non-conforming','Man;Or, in your own words:', 
                                     'Man;Or, in your own words:;Non-binary, genderqueer, or gender non-conforming'], 'Man')

df['Gender'] = df['Gender'].replace(['Woman;Non-binary, genderqueer, or gender non-conforming','Or, in your own words:;Woman;Non-binary, genderqueer, or gender non-conforming', 
                                     'Or, in your own words:;Woman'], 'Woman')

df['Gender'].unique()

df['Employment'].unique()

for i in df.Employment:
    df['Employment'] = df['Employment'].replace(i, str(i).split(';')[0])

df

df['DevType'].unique()


df.to_csv('surveyDFcleaned.csv')

profile = ProfileReport(df)
profile.to_notebook_iframe ()

