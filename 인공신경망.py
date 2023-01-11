import pandas as pd
import numpy as np
import Dir
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import to_categorica
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
df = pd.read_csv(Dir.dir+"[Dataset]_Module_18_(iris).data",header=None)
names = ["sepal_length", "sepal_width","petal_length", "petal_width", "class"]
df.columns = names
print(df.head())
print(df.info())
print(df.describe())

x_values = df[['sepal_length','sepal_width','petal_length','petal_width']]
print(x_values.head())
standardise = StandardScaler() # 표준척도는 분포의 평균값이 0이고 표준편차가 1이 되도록 데이터를 변환합니다.
x_values = standardise.fit_transform(x_values)
x_values_df = pd.DataFrame(x_values)

print(x_values_df.head())
print(x_values_df.describe())

# 신경망 모델을 초기화
model = Sequential()

# 6개의 노드가 있는 첫 번째 은닉 레이어를 추가합니다.
# Input_dim은 x_values 또는 입력 레이어의 수/특성 수를 나타냅니다.
# activation은 노드/뉴런이 활성화되는 방식을 나타냅니다. 우리는 relu를 사용할 것입니다. 다른 일반적인 활성화 방식은'sigmoid' 및 'tanh'입니다.
model.add(Dense(6, input_dim=4, activation='relu'))

# 6개의 노드가 있는 두 번째 은닉 레이어를 추가합니다.
model.add(Dense(6, activation='relu'))

# 3개의 노드가 있는 출력 레이어를 추가합니다.
# 사용된 activation은 'softmax'입니다. Softmax는 범주형 출력 또는 대상을 처리할 때 사용됩니다.
model.add(Dense(3,activation='softmax'))

# 모델을 컴파일합니다. optimizer는 모델 내에서 조정하는 방법을 의미합니다. loss은 예측된 출력과 실제 출력 간의 차이를 나타냅니다.
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# 각 클래스의 데이터 포인트 수 출력
print(df['class'].value_counts())

# 각기 다른 클래스에 대하여 다른 숫자를 지정한 딕셔너리
# 원-핫 인코딩으로 4개 열이 아닌 3개 열만 생성되도록 0부터 시작하는 값을 사용합니다.
label_encode = {"class": {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}}

# .replace를 사용하여 다른 클래스를 숫자로 변경
df.replace(label_encode,inplace=True)

# 각 클래스의 데이터 포인트 수를 출력하여 클래스가 숫자로 변경되었는지 확인
print(df['class'].value_counts())

# 클래스를 y_values로 추출
y_values = df['class']

# y_values 원-핫 인코딩
y_values = to_categorical(y_values)

print(y_values)

# x_values와 y_values로 모델을 훈련시킵니다.
# Epoch는 전체 데이터 세트가 모델을 학습하는 데 사용되는 횟수를 나타냅니다.
# Shuffle = True는 모델이 각 Epoch 후에 데이터 세트의 배열을 무작위로 지정하도록 지시합니다.
model.fit(x_values,y_values,epochs=20,shuffle=True)

# 신경망 모델을 초기화
model2 = Sequential()

# 6개의 노드가 있는 첫 번째 은닉 레이어를 추가합니다.
# Input_dim은 x_values 또는 입력 레이어의 수/특성 수를 나타냅니다.
# activation은 노드/뉴런이 활성화되는 방식을 나타냅니다. 우리는 relu를 사용할 것입니다. 다른 일반적인 활성화 방식은'sigmoid' 및 'tanh'입니다.
model2.add(Dense(6, input_dim=4, activation='relu'))

# 6개의 노드가 있는 두 번째 은닉 레이어를 추가합니다.
model2.add(Dense(6, activation='relu'))

model2.add(Dense(6, activation='relu'))
# 3개의 노드가 있는 출력 레이어를 추가합니다.
# 사용된 activation은 'softmax'입니다. Softmax는 범주형 출력 또는 대상을 처리할 때 사용됩니다.
model2.add(Dense(3,activation='softmax'))

model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model2.fit(x_values,y_values,epochs=20,shuffle=True)

model3 = Sequential()

# 모델을 컴파일합니다. optimizer는 모델 내에서 조정하는 방법을 의미합니다. loss은 예측된 출력과 실제 출력 간의 차이를 나타냅니다.

model3.add(Dense(6, input_dim=4, activation='relu'))

# 6개의 노드가 있는 두 번째 은닉 레이어를 추가합니다.
model3.add(Dense(8, activation='relu'))

model3.add(Dense(6, activation='relu'))
# 3개의 노드가 있는 출력 레이어를 추가합니다.
# 사용된 activation은 'softmax'입니다. Softmax는 범주형 출력 또는 대상을 처리할 때 사용됩니다.
model3.add(Dense(3,activation='softmax'))

# 모델을 컴파일합니다. optimizer는 모델 내에서 조정하는 방법을 의미합니다. loss은 예측된 출력과 실제 출력 간의 차이를 나타냅니다.
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model3.fit(x_values,y_values,epochs=20,shuffle=True)

# 신경망 모델을 초기화
model4 = Sequential()

# 6개의 노드가 있는 첫 번째 은닉 레이어를 추가합니다.
# Input_dim은 x_values 또는 입력 레이어의 수/특성 수를 나타냅니다.
# activation은 노드/뉴런이 활성화되는 방식을 나타냅니다. 우리는 relu를 사용할 것입니다. 다른 일반적인 활성화 방식은'sigmoid' 및 'tanh'입니다.
model4.add(Dense(6,input_dim=4,activation='relu'))

# 6개의 노드가 있는 두 번째 은닉 레이어를 추가합니다.
model4.add(Dense(6,activation='relu'))

# 3개의 노드가 있는 출력 레이어를 추가합니다.
# 사용된 activation은 'softmax'입니다. Softmax는 범주형 출력 또는 대상을 처리할 때 사용됩니다.
model4.add(Dense(3,activation='softmax'))

# 모델을 컴파일합니다. optimizer는 모델 내에서 조정하는 방법을 의미합니다. loss은 예측된 출력과 실제 출력 간의 차이를 나타냅니다.
model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# x_values와 y_values로 모델을 훈련시킵니다.
# Epoch는 전체 데이터 세트가 모델을 학습하는 데 사용되는 횟수를 나타냅니다.
# Shuffle = True는 모델이 각 Epoch 후에 데이터 세트의 배열을 무작위로 지정하도록 지시합니다.
model4.fit(x_values,y_values,epochs=200,shuffle=True)

# 데이터 프레임 df에서 원래 x_values를 추출합니다.
# 표준화는 모델이 학습할 데이터에만 기반해야 하므로 x_values를 다시 추출해야 합니다.
# 따라서 표준화하기 전에 먼저 데이터를 분할해야 합니다.
x_values = df[['sepal_length','sepal_width','petal_length','petal_width']]

# Test_size=0.25는 전체 데이터의 25%가 x_test 및 y_test로 배정되면75%는 x_train 및 y_train에 배정됨을 나타냅니다.
# random_state=10은 아래 코드를 실행할 때마다 분할이 동일하도록 하는 데 사용됩니다.
# 분할은 매번 랜덤으로 하기 때문입니다. 동일한 random_state는 매번 동일하게 분할되도록 보장하는 유일한 방법입니다.
x_train, x_test, y_train, y_test = train_test_split(x_values,y_values,test_size=0.25,random_state=10)

# x_train, x_test, y_train 및 y_test의 행 수 확인
print("Number of rows in x_train:", x_train.shape[0])
print("Number of rows in x_test:", x_test.shape[0])
print("Number of rows in y_train:", y_train.shape[0])
print("Number of rows in y_test:", y_test.shape[0])

# 이제 x 값을 표준화할 수 있습니다.
# StandardScaler 초기화
standardise = StandardScaler()

# .fit_transform을 사용하여 x_train 값을 표준화합니다.
x_train = standardise.fit_transform(x_train)

# .transform을 사용하여 x_test 값을 표준화합니다.
# 표준화는 x_train과 같아야 하므로 데이터를 맞출 필요가 없습니다.
x_test = standardise.transform(x_test)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_values,y_values,test_size=0.2,random_state=10)
# x_train, x_test, y_train 및 y_test의 행 수 확인
print("Number of rows in x_train:", x_train2.shape[0])
print("Number of rows in x_test:", x_test2.shape[0])
print("Number of rows in y_train:", y_train2.shape[0])
print("Number of rows in y_test:", y_test2.shape[0])

# 이제 x 값을 표준화할 수 있습니다.
# StandardScaler 초기화
standardise = StandardScaler()

# .fit_transform을 사용하여 x_train 값을 표준화합니다.
x_train2 = standardise.fit_transform(x_train2)

# .transform을 사용하여 x_test 값을 표준화합니다.
# 표준화는 x_train과 같아야 하므로 데이터를 맞출 필요가 없습니다.
x_test2 = standardise.transform(x_test2)

model_val = Sequential()

# 6개의 노드가 있는 첫 번째 은닉 레이어를 추가합니다.
# Input_dim은 x_values 또는 입력 레이어의 수/특성 수를 나타냅니다.
# activation은 노드/뉴런이 활성화되는 방식을 나타냅니다. 우리는 relu를 사용할 것입니다. 다른 일반적인 활성화 방식은'sigmoid' 및 'tanh'입니다.
model_val.add(Dense(6,input_dim=4,activation='relu'))

# 6개의 노드가 있는 두 번째 은닉 레이어를 추가합니다.
model_val.add(Dense(6,activation='relu'))

# 3개의 노드가 있는 출력 레이어를 추가합니다.
# 사용된 activation은 'softmax'입니다. Softmax는 범주형 출력 또는 대상을 처리할 때 사용됩니다.
model_val.add(Dense(3,activation='softmax'))

# 모델을 컴파일합니다. optimizer는 모델 내에서 조정하는 방법을 의미합니다. loss은 예측된 출력과 실제 출력 간의 차이를 나타냅니다.
model_val.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model_val.summary())
# x_values와 y_values로 모델을 훈련시킵니다.
# Epoch는 전체 데이터 세트가 모델을 학습하는 데 사용되는 횟수를 나타냅니다.
# Shuffle = True는 모델이 각 Epoch 후에 데이터 세트의 배열을 무작위로 지정하도록 지시합니다.
model_val.fit(x_train,y_train,epochs=50,shuffle=True, validation_data=(x_test,y_test))

df2 = pd.read_csv(Dir.dir+"[Dataset]_Module_18_(iris).data",header=None)
names = ["sepal_length", "sepal_width","petal_length", "petal_width","class"]
df2.columns = names

x_new = df2[['sepal_length','sepal_width','petal_length','petal_width']]
x_new_scale = standardise.transform(x_new)
y_new = model_val.predict(x_new_scale)
print(y_new)
print(y_new.shape)

flower_types = []
for ii in range(0,y_new.shape[0]):
    flower_types.append(np.argmax(y_new[ii,:]))

label_encode = {"class": {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}}
df3 = pd.DataFrame(flower_types)
df3.replace(label_encode,inplace=True)

print(flower_types)
print(df3)