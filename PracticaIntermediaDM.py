import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np


df = pd.read_csv('survey_results_public.csv')
df.head()

#problema 1
#slice de dataframe original filas que no tengan valores nulos
filter_df = df['Gender'].notna() & df['ConvertedComp'].notna() #no nulos ni en genero ni en salario
new_df = df[filter_df]
fillter_df = (new_df['Gender']=='Woman')| (new_df['Gender']=='Woman;Man') #sin nulos en gender y salario
new_df[filter_df]
def sub_str(str_,  sub_str_):
        return sub_str_  in str_

def uniques(col):
        l = list( col.unique())
        l = ';'.join(l).split(';')
        return list(set(l))

def filter_not_nulls(df, *cols):
    #obtiene nombre de las columnas
    #regresafiltrado el data frame
    f = df[cols[0]].notnull()
    for col in cols[1:]:
        f = f & df[col].notnull()
    return df[f]
    
def five_numbers_summary(col):
    mini = col.min()
    maxi = col.max()
    ql1     = col.quantile(0.25)
    ql2    = col.quantile(0.5)
    ql3    = col.quantile(0.75)
    return mini, maxi, ql1, ql2, ql3

def std_mean(col):
    return col.std(), col.mean()
    

#Filtrar los no nulos
filter_df = df['Gender'].notna() & df['ConvertedComp'].notna() #no nulos ni en genero ni en salario
new_df = df[filter_df]
#filtramos todas las mujeres 
filter_df = new_df['Gender'].apply(sub_str, args=('Woman',))
new_df[filter_df]

uniques(new_df['Gender'])
new_df = filter_not_nulls(df, 'Gender', 'ConvertedComp')
uniques_= uniques(new_df['Gender']) #uniques almaena los valores diferentes de genero
d_genders = {} #diccionario que tendra como key los valores de (man, woman ...)
                            #y como valores el dataframa filtradi para cada genero
for uni in uniques_: 
    #Filtar por cada genero
    #print(uni)
    filter_df = new_df['Gender'].apply(sub_str, args=(uni,))
    #Dataframe filtrado
    f_df = new_df[filter_df]
    #Asignamos dtaframe filtrado a cada genero
    d_genders[uni] = f_df
    #d_gender ['Woman'], head()

plt.figure(num = None , figsize = (8, 4 ), dpi = 80, facecolor = '#FEF785', edgecolor='k')
s = ' : Min = {0}, Max = {1}, Q1 = {2}, Median = {3}, Q3 = {4}, Std= {5}, Mean ={6}'
i = 1
for gender, g_df in d_genders.items():
    print(gender[:10], s.format(*five_numbers_summary(g_df['ConvertedComp']) , *std_mean(g_df['ConvertedComp'])))
    plt.subplot(1,  len(d_genders), i)
    plt.boxplot(g_df['ConvertedComp'], sym = '')
    plt.ylabel('Amount')
    plt.xlabel( gender[:10])
    plt.title(f'Boxplot for {gender[:10]}')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
    
    i += 1
plt.savefig("gender.png")


plt.figure(num = None , figsize = (8, 4 ), dpi = 80, facecolor = '#FEF785', edgecolor='k')
s = ' : Min = {0}, Max = {1}, Q1 = {2}, Median = {3}, Q3 = {4}, Std= {5}, Mean ={6}'
i = 1
for gender, g_df in d_genders.items():
    print(gender[:10], s.format(*five_numbers_summary(g_df['ConvertedComp']) , *std_mean(g_df['ConvertedComp'])))
    plt.subplot(1,  len(d_genders), i)
    plt.boxplot(g_df['ConvertedComp'], sym = '')
    plt.ylabel('Amount')
    plt.xlabel( gender[:10])
    plt.title(f'Boxplot for {gender[:10]}')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
    
    i += 1
plt.savefig("gender.png")

#Problema 2
plt.figure(figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
boxplots(df, 'ConvertedComp', 'Ethnicity', ncols = 3, nrows =3, truncate = 15, filename = 'Etnia.png')
plt.savefig("Etnia.png")
plt.show()

#Problema 3
plt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
boxplots(df, 'ConvertedComp', 'DevType', ncols = 6, nrows =4 , truncate=15)

#problema 4
col_a = 'ConvertedComp'
col_b = 'Country'
s = '{0}: Mean = {1}, Median = {2}, Std = {3}'
new_df = filter_not_nulls(df, col_b, col_a) # Filtra para las columnas gender y salary
countries = uniques(new_df[col_b]) # uniques almacenará los valores diferentes de género
for country in countries:
    filter_df = new_df['Country'] == country  # 
    f_df = new_df[filter_df]                  # Dataframe filtrado para cada país
    print(s.format(
                    country, 
                    f_df['ConvertedComp'].mean(),
                    f_df['ConvertedComp'].median(),
                    f_df['ConvertedComp'].std()
                  ))

#problema 5
new_df     =  filter_not_nulls(df, 'DevType')
devtypes  = uniques(new_df['DevType'])
devtypes
freqs = {} #key = devtypes, value =freq
for devtype in devtypes:
    freq = sum(new_df['DevType'].apply(sub_str, args =( devtype, )))
    freqs[devtype]=freq
height =np.arange(len(freqs))
plt.figure(figsize= (12, 10))
plt.bar(height=list(freqs.values()), x=height)
plt.xticks(height, freqs.keys(), rotation = 90)



#Problema 6
from textwrap import wrap
df['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
df['YearsCode'].replace('More than 50 years', '51', inplace=True)
df['YearsCode'] =  df['YearsCode'].astype('float64') # Convierte la columna a flotantes
def hist(df, col_a, col_b, nrows=1, ncols=None,
        xlabel=None, ylabel=None, filename=None, 
        nbins=10, fontsize=12):
    new_df = filter_not_nulls(df, col_b, col_a)
    uniques_ = uniques(new_df[col_b])
    if not ncols:
        ncols = len(uniques_)
    if ylabel is None:
        ylabel = "Amount"
    if not xlabel:
        xlabel = col_a
    for i, unique in enumerate(uniques_):
        filter_df = new_df[col_b].apply(sub_str, args=(unique,))
        f_df = new_df[filter_df]
        plt.subplot(nrows,ncols,i+1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title('\n'.join(wrap(unique,30)), fontsize=fontsize)
        plt.hist(f_df[col_a], bins=nbins)
        plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
plt.figure(figsize=(4,6), facecolor='w')
hist(df, 'YearsCode', 'Gender', xlabel='Experience', ylabel='',
    nrows=3, ncols=1, nbins=10)

#Problema 7
new_df = filter_not_nulls(df, 'WorkWeekHrs', 'DevType') # Limpiamos los nulos
filter_df = ((new_df['WorkWeekHrs'] < 80) & 
             (new_df['WorkWeekHrs'] > 30))              # Eliminamos pocas y muchas horas
new_df = new_df[filter_df]                              # Sin valores nulos
                                                        # Sin valores "no posibles"
# figsize = (width, height)
plt.figure(figsize=(8,16), facecolor='w')
hist(new_df, 'WorkWeekHrs', 'DevType', 
     xlabel='Worked hours per week', ylabel='',
     nrows=8, ncols=3, nbins=10)

#Problema 8
new_df = filter_not_nulls(df, 'Age', 'Gender') # Limpiamos los nulos
filter_df = ((new_df['Age'] < 80) & 
             (new_df['Age'] > 10))           # Eliminamos pocas y muchas horas
new_df = new_df[filter_df]                              # Sin valores nulos
                                                        # Sin valores "no posibles"
# figsize = (width, height)
plt.figure(figsize=(8,4), facecolor='w')
hist(new_df, 'Age', 'Gender', 
     xlabel='Age', ylabel='',
     nrows=1, ncols=3, nbins=50)
#Problema 9
df['LanguageWorkedWith'].replace('')
new_df = filter_not_nulls(df, 'Age', 'LanguageWorkedWith') # Limpiamos los nulos
languages = uniques(new_df['LanguageWorkedWith'])
# s = '{0}: Mean = {1}, Median = {2}, Std = {3}'
s = '%s: Mean = %.3f, Median = %.3f, Std = %.3f'
for language in languages:
#     print(language)
    filter_df = new_df['LanguageWorkedWith'].apply(sub_str, args=(language,))
#     print(sum(filter_df))
    f_df = new_df[filter_df]['Age'] # Dataframe filtrado por lenguage
#     print(s.format(language, 
#                    f_df.mean(), 
#                    f_df.median(), 
#                    f_df.std()
#                   )
#          )
    print(s % (
                language,
                f_df.mean(), 
                f_df.median(), 
                f_df.std()
              )
         )
#Problema 10
new_df = filter_not_nulls(df, 'ConvertedComp', 'YearsCode')
x = new_df['ConvertedComp'].to_numpy()
y = new_df['YearsCode'].to_numpy()
corr = np.corrcoef(x=x, y=y)
print(corr)
plt.figure(figsize=(9,9), facecolor='w')
plt.scatter(x=x, y=y)
plt.title('Salary and Experience correlation')
plt.xlabel('Salary')
plt.ylabel('Experience')

#problema 11
new_df = filter_not_nulls(df, 'ConvertedComp', 'Age')
x = new_df['ConvertedComp'].to_numpy()
y = new_df['Age'].to_numpy()
corr = np.corrcoef(x=x, y=y)
print(corr)
plt.figure(figsize=(9,9), facecolor='w')
plt.scatter(x=x, y=y)
plt.title('Salary and Age correlation')
plt.xlabel('Salary')
plt.ylabel('Age')

# Problema 12
edlevels ={
    'I never completed any formal education' : 0,
    'Primary/elementary school' : 1,
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 2,
    'Associate degree': 3,
    'Professional degree (JD, MD, etc.)' : 4,
    'Bachelor’s degree (BA, BS, B.Eng., etc.)' : 4,
    'Some college/university study without earning a degree' : 5,
     'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 6,
     'Other doctoral degree (Ph.D, Ed.D., etc.)': 7,
 }
def edlevel(item):
    return edlevels[item]
new_df = filter_not_nulls(df, 'EdLevel', 'ConvertedComp')
y = new_df['EdLevel'].apply(edlevel)
x = new_df['ConvertedComp'].to_numpy()
corr = np.corrcoef(x=x, y=y)
edlevels ={
    'I never completed any formal education' : 0,
    'Primary/elementary school' : 1,
    'Secondary school' : 2,
    'Associate degree': 3,
    'Professional degree ' : 4,
    'University without a degree' : 5,
    'Master’s degree': 6,
    'Other doctoral degree': 7,
 }
print(corr)
plt.figure(figsize=(9,9), facecolor='w')
plt.scatter(x=x, y=y)
plt.title('Salary and Edcuation  correlation')
plt.yticks(list(edlevels.values()), list(edlevels.keys()), rotation=0)
plt.ylabel('Education level')
plt.xlabel('Salary')

#Problema 13
new_df   = filter_not_nulls(df, 'LanguageWorkedWith') # new_df is indexable
devtypes = uniques(new_df['LanguageWorkedWith'])
freqs = {} # keys = devtypes, values = Frec
for devtype in devtypes:
    freq = sum(new_df['LanguageWorkedWith'].apply(sub_str, args=(devtype,)))
    freqs[devtype] = freq
freqs = sorted(freqs.items(), key=lambda item : item[1], reverse=True)
freqs = {k:v for k, v in freqs}
x = np.arange(len(freqs))
plt.figure(figsize=(12,10), facecolor='w')
plt.bar(height=freqs.values(), x=x)
plt.xticks(x, freqs.keys(), rotation=90)
