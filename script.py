import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import resample

df = pd.read_csv("./carclaims.csv")

df.head()

# print("Количество признаков изначально:",df.shape[1])
# print("Количество примеров:",df.shape[0])

print("Пропуски в данных не обнаружены")
print('-'*50)
# print(df.isnull().sum())

plt.figure(figsize=(10,8))
plt.pie(df.FraudFound.value_counts().values,labels=['Нет', 'Да'],  autopct='%.0f%%')
plt.title("Признак мошенничества")
# plt.show()

df.loc[df['FraudFound'] == 'No','FraudFound'] = 0
df.loc[df['FraudFound'] == 'Yes','FraudFound'] = 1

df['FraudFound'] = df['FraudFound'].astype(int)

policyAge = df.groupby('AgeOfPolicyHolder')['FraudFound'].sum()
plt.figure(figsize=(20,8))
plt.title("Зависимость количества мошенничеств от возраста держателя полиса")

ax = sns.barplot(x=policyAge.index,y=policyAge.values)
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x()+0.24, p.get_height()*1.01))
plt.xlabel("Возраст держателя полиса")
plt.ylabel("Количество случаев мошенничества")
plt.yticks([])
# plt.show()

gender = df.groupby('Sex')['FraudFound'].sum()
plt.figure(figsize=(10,8))
plt.title("Диаграмма распределения количества мошенничеств от пола держателя полиса")

plt.pie(gender.values, labels=['Женщины', 'Мужчины'],  autopct='%.0f%%')

# plt.show()

accidentArea = df.groupby('AccidentArea')['FraudFound'].sum()

plt.figure(figsize=(10,8))
plt.title("Диаграмма распределения количества мошенничества от области происшествия")

plt.pie(accidentArea.values,labels=['Село', 'Город'],  autopct='%.0f%%')

# plt.show()

fault = df.groupby('Fault')['FraudFound'].sum()

plt.figure(figsize=(10,8))
plt.title("Диаграмма распределения виновников происшествий")
plt.pie(fault.values,labels=['Держатели полисов', 'Третьи лица'],  autopct='%.0f%%')
# plt.show()

cars = df.groupby('NumberOfCars')['FraudFound'].sum()
plt.figure(figsize=(20,8))
plt.title("Зависимость мошенничества от числа автомобилей,указанных в полисе")

ax = sns.barplot(x=cars.index,y=cars.values)
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x()+0.4, p.get_height()*1.01))
plt.xlabel("Число автомобилей указанных в полисе")
plt.ylabel("Мошенничество")
plt.yticks([])
# plt.show()

fraud = df[df['FraudFound'] == 1]
plt.figure(figsize=(10,5))
plt.title("Зависимость мошенничества от семейного статуса")
sns.countplot(x=fraud['MaritalStatus'])
plt.xlabel("Семейный статус")
plt.ylabel("Мошенничество")

le = LabelEncoder()

cols = df.select_dtypes('O').columns

df[cols]= df[cols].apply(le.fit_transform)
df['Year'] = le.fit_transform(df.Year)

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),cmap="YlGnBu")
plt.title("Тепловая карта корреляции")
# plt.show()

df_new = df[['AccidentArea','Sex',\
       'MaritalStatus','Fault','Year',\
       'DriverRating', 'Days:Policy-Accident', 'Days:Policy-Claim',\
       'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',\
       'PoliceReportFiled', 'WitnessPresent', 'AgentType',\
       'NumberOfSuppliments', 'AddressChange-Claim', 'NumberOfCars',\
       'BasePolicy', 'FraudFound']]

plt.figure(figsize=(20,10))
sns.heatmap(df_new.corr(),cmap="YlGnBu")
plt.title("Тепловая карта корреляции")
# plt.show()

def conf_matrix(y_test,y_pred):
    con_matrix = confusion_matrix(y_test,y_pred)
    con_matrix = pd.DataFrame(con_matrix,range(2),range(2))
    
    plt.figure(figsize=(5,5))
    plt.title("Confusion Matrix")
    sns.heatmap(con_matrix,annot=True,cbar=False,fmt='g')

X = df_new.drop('FraudFound',axis=1)
y = df_new[['FraudFound']]
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

n = df_new.FraudFound.value_counts()[0]

df_majority = df_new[df_new.FraudFound==0]
df_minority = df_new[df_new.FraudFound==1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples = n,random_state=42)

df_upsampled = pd.concat([df_majority,df_minority_upsampled])
df_upsampled.FraudFound.value_counts()

X = df_upsampled.drop('FraudFound',axis=1)
y = df_upsampled[['FraudFound']]
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y, random_state=42)

print(f'Набор данных для обучение содержит = {X_train.shape[0]} строк и {X_train.shape[1]} признаков')
print(f'Набор данных для тестирования содержит = {X_test.shape[0]} строк и {X_test.shape[1]} признаков')
print('-'*50)
print(f'Набор данных сбалансирован по классам и содержит = {n} строк для класса Мошенничество и \
{n} строк для класса НЕ Мошенничество')
print('-'*50)
print(f'Обучен и протестирован алгоритм Случайный Лес\n')

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_upscale_pred = rfc.predict(X_test)

acc_rfc_upscale=accuracy_score(y_test, rfc_upscale_pred)
print(f"Точность классификации:\t\t{acc_rfc_upscale*100:.0f}%\n")
conf_matrix(y_test,rfc_upscale_pred)