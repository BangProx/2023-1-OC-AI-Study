# ML 실습

# 1. 간단한 분류 프로젝트 해보기

머신러닝은 데이터에서 정해진 패턴을 파악하는 것이다. 데이터에서 어떤 대상(Target)과 대상이 가지는
특징(Feature)들의 관계를 파악하는 것이다. 

<aside>
📌 내가 예전에 학습한 데이터에 의하면... 부리의 길이가 x'고 두께는 y'고 몸무게는 z', 사는 지역은 k' 이니까.... 이 펭귄의 종은 A겠군!

</aside>

## 1-1 데이터 불러오기

유방암의 여부를 예측해보는 task를 진행하자

```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

dataset = pd.concat([X, y], axis=1)
dataset
```

> `load_breast_cancer()`의 매개변수는 `return_X_y` 와 `as_frame`이 있다.
`return_X_y` : 부울 값을 전달해야한다. 디폴트값은 False이다. 만약 True이면 Bunch 객체 대신에 (data, target)을 반환한다. 
`as_frame` : 부울 값을 전달해야한다. True인 경우 data는 pandas DataFrame이다.
Target은 target의 컬럼의 개수에 따라 pandas DataFrame 이나 Series가 된다.
`return_X_y` 가 True인 경우 (data, target)은 pandas DataFrame이나 Series가 된다.
> 

<aside>
📌 만약 에러가 발생하면 본인의 컴퓨터에 sklearn 패키지가 다운로드 되어있는지 확인해야한다.
다운로드되어있지 않다면 `pip install scikit-learn` 명령어로 다운로드 받을 수 있다.

</aside>

> 이 데이터는 환자의 특성을 분석해서 유방암 양성 여부를 파악하기 위해 제작된 데이터이다. 
자료는 반지름, 오목도, 뻑뻑함등의 지표에 대해 평균값과 최악인 값이다. 결과는 target이 0인 경우 양성, 1인 경우 음성이다.
> 

## 1-2 train_test_split

Dataset을 나눌 때 test_size 옵션으로 Train, Test의 비율을 설정할 수 있고, random_state로 seed 값을 지정할 수 있다. 자세한 설명은 → [설명링크](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

아래 예시 코드의 빈 칸에 들어갈 함수를 입력하고, 학습 데이터와 검증 데이터를 나눠주세요!

```python
**from** sklearn.model_selection **import** train_test_split

X_train, X_val, y_train, y_val **=** """HERE YOUR CODE"""
print(f"Shape of X_train: {X_train**.**shape}")
print(f"Shape of y_train: {y_train**.**shape}")
print(f"Shape of X_val: {X_test**.**shape}")
print(f"Shape of y_val: {y_test**.**shape}")
```

**단, 검증데이터의 비율은 0.2, random_state=42로 고정시켜주세요.**

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, random_state = 42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_val: {y_val.shape}")
```

> 앞서서 유방암 관련 데이터를 (data, target)으로 따로 X, y에 저장했다. 따라서, train_test_split에 
X,y를 전달하고 test_size = 0.2, random_state = 42로 설정했다.
> 

# 2. Preprocessing 전처리

## **2-1 null값 삭제**

**우리가 사용하는 데이터에는 결측치가 없기 때문에, 일단 나름대로 Data Frame을 생성해보고, 결측치를 제거하는 코드를 실습해볼게요!**

```python
**import** pandas **as** pd
**import** numpy **as** np

df **=** pd**.**DataFrame(np**.**random**.**randn(100, 5), columns**=**["A","B","C","D","E"])

**for** _ **in** range(10):
    row_idx **=** np**.**random**.**choice(df**.**index)
    col_idx **=** np**.**random**.**choice(df**.**columns)
    df**.**loc[row_idx, col_idx] **=** np**.**nan

print(f"🔎 # of NaN Values :\n {df**.**isnull()**.**sum()}")
print(f"Shape of Data Frame : {df**.**shape}")

**>** HERE YOUR CODE!

print(f"🚀 결측치 처리 후 :\n {df**.**isnull()**.**sum()}")
print(f"Shape of Data Frame : {df**.**shape}")
```

검색 힌트 : pd.dropna(), pd.isnull().sum(), [문자열 포맷팅](https://hyjykelly.tistory.com/65)
random.choice()는 전달된 리스트/튜플에서 랜덤으로 하나의 요소를 선택한다.

```python
# > HERE YOUR CODE 부분에 들어갈 코드
df.dropna(inplace = True);
```

> for문은 DataFrame에서 특정 행과 열을 전달받아서 해당 행과 열에 해당하는 요소를 nan값으로 치환한다. 따라서, DataFrame의 랜덤한 10개의 요소가 nan값이 되었다.
그 다음에는 현재 결측치의 총 개수와 DataFrame의 shape를 출력하였다.
결측치를 제거하기위해 판다스 `dropna()` 메서드를 활용했고 `inplace = True` 로 옵션값을 설정해서 수정사항을 기존 DataFrame에 반영했다.
> 

## **2-2 Scaling 하기**

### 문제

`sklearn`의 `StandardScaler`를 이용하여 `X_train`과 `X_val`을 Scaling 해주세요.

단, X_train엔 `fit_transform()`을, X_val엔 `transform()` 메소드를 적용해서 Scaling 해주세요!

그리고 왜 학습데이터엔 `fit_transform()`을 사용해도 되지만, 검증용이나 테스트 데이터엔 `transform()`만을 이용해야 하는지 적어주세요!

Base Line

```python
**from** sklearn.preprocessing **import** StandardScaler
**import** matplotlib.pyplot **as** plt

scaler **=** '''HERE YOUR CODE!'''

scaled_X_train **=** '''HERE YOUR CODE!'''
scaled_X_val **=** '''HERE YOUR CODE!'''

scaled_X_train_check **=** scaled_X_train**.**reshape(30, **-**1)
print(f"Scaling전 데이터의 최대, 최소, 평균, std: {X_train['mean texture']**.**max(), X_train['mean texture']**.**min(),  X_train['mean texture']**.**mean(),  X_train['mean texture']**.**std()}")
print(f"Scaling후 데이터의 최대, 최소, 평균, std: {scaled_X_train_check[0]**.**max(), scaled_X_train_check[0]**.**min(), scaled_X_train_check[0]**.**mean(), scaled_X_train_check[0]**.**std()}")
```

검색 힌트 : fit_transform()과 transform()차이, sklearn Standard Scaler, sklearn Scaler 종류

```python
# 1)
scaler = StandardScaler()
# 2)
X_train_scaled = scaler.fit_transform(X_train)
# 3)
X_val_scaled = scaler.transform(X_val)
```

> reshape(30, -1)은 앞의 nparray를 30 x n 행렬로 변환해준다. 여기서 두번째 매개변수에 -1을 전달하면서 자동으로 뒤의 n을 완성해준다. 예를 들어 길이가 90인 nparray를 reshape(30,-1)로 하면
30 x 3 크기의 행렬로 자동으로 변환된다. 행의 길이가 30인 이유는 항목이 30개이기 때문이다.
scaler는 StandardScaler로 진행했고 train data set은 fit_transform()으로 스케일링하고 
test data set은 transform()을 통해 스케일링을 진행했다.
> 

## **2-3 학습시키기**

이제 우리가 전처리한 데이터를 학습시키는 일만 남았습니다.

데이터를 학습할 모델을 고르는 것도 사실 중요한 일 중 하나지만, 일단 저희는 Tree 모델 중에서 가~~~~장 기본이 되는 `DecisionTree`와 `Random Foreset`를 사용해서 성능을 비교해볼게요!

이번 문제는 그냥 간단하게, `아~ 이런 이런 함수를 써서 학습시키고 예측하는구나~` 정도로만 알고 넘어가셔도 좋을 것같습니다.

sklearn의 `DecisionTreeClassifier`로 `scaled_X_train`과 `y_train`을 이용해서 학습을 진행해주세요!

단, `DecisionTreeClassifier`의 `random_state`는 42로 고정시켜주세요.

```python
**from** sklearn.tree **import** DecisionTreeClassifier

classifier **=** '''HERE YOUR CODE!!'''

classifier**.**'''HERE YOUR CODE!!'''

print("🤖Training is Done!")
```

검색 힌트: sklearn DecisionTree, sklearn 모델 학습

```python
classifier = DecisionTreeClassifier(random_state = 42)
classifier.fit(scaled_X_train,y_train)
print("🤖Training is Done!")
```

> DecisionTreeClassifier의 랜덤시드 값을 42로 고정했고, fit() 메서드를 통해 scaled_X_train 값과 y_train값으로 학습을 완료했다.
> 

## **2-4 예측하기**

2-3에서 모델을 우리 데이터에 맞게 학습시켰습니다. 이제 그럼 이 학습된 모델이 실전에서 잘 작동할 수 있는지 확인해봐야겠죠?

학습된 모델을 기반으로 데이터를 넣어 prediction을 구할 수 있습니다.

2-4에서 학습시킨 모델로 `scaled_X_val`을 예측해주세요!

그리고 모델의 정확도(Accuracy)가 얼마나 되는지, `sklearn`의 `accuracy_score`를 통해 계산해주세요!

```python
**from** sklearn.metrics **import** '''HERE YOUR CODE!!'''

predictions **=** classifier**.**'''HERE YOUR CODE!!'''
accuracy **=** '''HERE YOUR CODE!!'''

print(f"Model Accuracy: {accuracy}")
```

검색 힌트: sklearn 모델 predict, sklearn 정확도 계산

```python
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_val)
print(f"Model Accuracy: {accuracy_score(y_val,y_pred)}")
#결과값
#Model Accuracy: 0.8596491228070176
```

> 얼래리요 근데 왜 정확도가 제시된거랑 차이가 나지? 랜덤 시드 잘못 설정했나? 아닌데?
그 이유는 바로 예측할때 전달한 테스트 셋이 스케일되지 않았기 때문이다. 이래서 스케일링 하는구나!!
> 

```python
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(scaled_X_val)
accuracy = accuracy_score(y_val,y_pred)
print(f"Model Accuracy: {accuracy}")
#결과값
#Model Accuracy: 0.9473684210526315
```

> 스케일된 테스트 셋을 전달하자 정확도가 제대로 나왔다. `y_pred`는 스케일된 셋을 기반으로 예측한 
결과이고`y_val`은 정답이다. sklearn의 `metrics`의 `accuracy_score()` 함수를 이용해서 정확도를 계산할 수 있다.
> 

## **2-5 다른 모델도 써보기**

우리는 `DecisionTree`알고리즘을 통해 94% 라는 높은 정확도를 얻어냈습니다.

그럼 Ensemble 모델의 원조할머니급인 `Random Forest`의 성능도 한번 확인해볼까요?

랜덤포레스트로 학습과 예측, 정확도 계산까지의 코드를 완성해주세요!

단, `RandomForestClassifier`의 random_state는 42로 고정해주세요.

```python
**from** sklearn.ensemble **import** '''HERE YOUR CODE!!'''

rf_clf **=** '''HERE YOUR CODE!!'''
rf_clf**.**'''HERE YOUR CODE!!'''
rf_prediction **=** rf_clf**.**'''HERE YOUR CODE!!'''

rf_acc **=** '''HERE YOUR CODE!!'''
print(f"Random Forest Model Accuracy: {rf_acc}")
```

검색 힌트: sklearn Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 42)
rf_clf.fit(scaled_X_train,y_train)
rf_prediction = rf_clf.predict(scaled_X_val)
rf_acc = accuracy_score(y_val,rf_prediction)
print(f"Random Forest Model Accuracy : {rf_acc}")
# 결과
# Random Forest Model Accuracy : 0.9649122807017544
```

> `rf_clf`에 `RandomForestClassifier()`로 classifer(분류기)를 지정해줬다. 이후 scaled된 train data로 학습을 마쳤고 scaled된 test data로 예측을 진행후 정확도 분석을 했다.
Random Forest가 Decision Tree보다 조금 더 정확하다는 것을 확인할 수 있었다.
> 

## 심화 학습

우리는 방금 Decision Tree와 Random Forest의 결과를 비교해봤습니다. 분명 같은 데이터로 두 모델을 학습시켰는데, Random Forest의 정확도가 조금 더 높게 나왔습니다.

**왜 그 럴 까 요?**

Random Forest는 이름에서도 느껴졌듯이, Tree가 여러개 모인 모델입니다.

하나보단 둘이 더 낫고, 셋보단 넷, 넷보단 여러명이 더 나을 때가 있죠. 우리가 함께 모여 팀프로젝트를 하는 이유가 바로 그거구요.

---

> 기존의 결정 트리는 해당 데이터에 맞춰서 분류를 진행했기 때문에 과적합 문제가 자주 발생했다. 
하지만, Random Forest는 앙상블 기법을 통해 훈련 과정에서 구성한 여러 개의 결정 트리들을 랜덤하게 학습시켜 결과 도출에 활용했다. 즉, 특정 특성을 선택하는 트리를 여러개 생성해서 이들을 
기반으로 작업을 수행하고 각각의 트리가 독립적으로 학습하기 때문에 학습과정을 병렬화할 수 있다.
일반적인 Decision Tree는 Tree correlation이라고 하는 특정 feature 하나가 정답에 많은 영향을
주게 되면 대부분의 결과치가 비슷하게 나오는 문제가 있었다. 하지만 Random Forest는 해당 문제를 해결했다.
>