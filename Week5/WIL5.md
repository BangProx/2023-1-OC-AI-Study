# 5주차 : ML 기초

# 부제 : sklearn 활용하기

이번 주차에서는 코드 그 자체보다는 개념의 흐름에 집중을 해야한다. 결국 나중에 작성하는 딥러닝 코드들도 결국 이런 흐름을 코드로 표현한 것에 불과하다. 

# 1. 데이터 불러오기

# 2. 데이터셋 분리(train_test_split)

### 개요

머신러닝을 훈련시킬 때는 머신러닝이 패턴을 학습할 **학습데이터**와 모델의 성능을 학습과정에서 평가할 
**검증데이터**, 마지막으로 실전 투입시 성능을 평가하기위한 **테스트데이터**가 필요하다. 실전에선, 데이터 한 뭉텅이만 있고, 우리는 그 데이터에 대해 학습시킨 다음, 미래에 들어올 미지의 데이터를 예측해야한다. 즉, 환자 500명의 데이터만을 가지고, 앞으로 들어올 환자(기존의 500명에는 포함 안 되어 있는 새로운 환자)의 데이터만을 가지고 유방암이 걸렸는지 안 걸렸는지 예측을 해내야 한다는 것이다. 그래서 가지고 있는 데이터를 **분리**시켜야 한다. 이 모델이 어느 정도의 성능이 될 지 미리 알아둘 필요가 있기 때문이다. `sklearn`의 `train_test_split`을 이용하면, 쉽게 처리할 수 있다.

- train/test를 분리하는 이유
    
    <aside>
    📌 train/test 또는 train/validation 으로 구분을 하는 이유는 머신러닝 모델에 train데이터를 
    100% 학습시킨 이후에 test 데이터에 모델을 적용시켰을때 성능이 안나오는 경우가 많기
    때문이다. 이런 현상을 **overfitting** 되었다고 하는데 말 그대로 모델이 제공된 데이터셋에만 
    과도하게 적합하게 학습된 나머지 조금이라도 벗어난 케이스에 대해서 예측율이 현저하게 
    떨어지기 때문이다. 따라서, **overfitting**을 방지하는게 모델 성능에 매우 중요하다.
    따라서, 기존 train/test로 나뉘어있던 데이터 셋에서 train을 train/validation으로 일정 비율을 쪼갠다. 학습시에는 train 셋으로, 학습 후 중간중간 validation 셋으로 내가 학습한 
    모델 평가를 해주는 것이다.
    모델이 과적합되었다면, validation 셋으로 검증시 예측율이나 오차율이 떨어지는 현상이 확인된다. 이런 현상이 일어나면 학습을 종료한다.
    
    </aside>
    

```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, 
train_size=None, random_state=None, shuffle=True, stratify=None)
```

### Parameter

**arrays** : 같은 길이의 연속된 indexable*(데이터를 인덱싱할 수 있는 객체들*) 즉, 분할시킬 데이터를 입력한다. (입력한 데이터 타입은 Python list, Numpy array, scipy-sparse matrices, Pandas dataframe이다.)

**test_size** : 테스트 데이터셋의 비율(float)이나 갯수(int) *(default = None)* train_size에 반대되는 옵션.
**float : 0.0~1.0 사이의 값이어야하고 데이터 셋의 비율이다.
int :  테스트 샘플의 절대값에 해당하는 수이다. 
None : train_size와의 보수로 설정된다. 즉 train_size + test_size = 1.0
train_size의 값도 None이면 0.25로 지정된다.

**train_size** : 학습 데이터셋의 비율(float)이나 갯수(int) *(default = test_size의 나머지)*
나머지 float, int에 대한 내용은 `test_size`와 동일하다

**random_state** : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 *(int나 RandomState로 입력)*

**shuffle** : 셔플여부설정하는 옵션으로 부울값을 전달해야한다. *(default = True)*
`shuffle = False`이면 `stratify`는 None이어야 한다.

**stratify** : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다. classification을 다룰 때 매우 중요한 옵션값으로 stratify 값을 target으로 지정해주면 각각의 **class 비율(ratio)을 train / validation에 유지**해 준다. (한 쪽에 **쏠려서 분배되는 것을 방지**) 만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있다.

### Return

**X_train, X_test, Y_train, Y_test** : 
arrays에 데이터와 레이블을 둘 다 넣었을 경우의 반환이며, 데이터와 레이블의 순서쌍은 유지된다.

**X_train, X_test** : 
arrays에 레이블 없이 데이터만 넣었을 경우의 반환.

# 3. Preprocessing(전처리)

데이터 셋 분리에 성공했다면, 전처리를 해야한다. 전처리는 모델의 성능을 가르는 가장 중요한 요소이다.
일단 ML기초 이므로 간단하게 결측치 제거와 Scaling만 진행한다.

## 1) 결측치 제거

데이터를 학습하는데 있어서 null값(결측치)는 도움이 되지 않는다. 오히려 학습을 방해한다. 결측치를 다루는 방법은 다양하지만 크게 두가지로 분류할 수 있다.

1. 삭제하기
2. 다른 값으로 대체하기
    1. 통계값(평균값, 최빈값, 중앙값..)
    2. EDA를 통해 알아낸 사실로 대체하기

## 2) Scaling

모델의 안정적인 학습을 위해서는 데이터의 단위를 통일시키는 것이 좋다. 그 이유는 아래와 같다.

1. Feature 간의 상대적 중요도가 잘 드러난다.
2. 이상치의 영향력이 감소한다.
3. 연산 효율이 향상된다.

Scaling을 하기 위해 Scaler를 사용한다. 

### Scaler의 종류

1. Standard Scaler
    
    기존의 변수의 범위를 정규 분포로 변환시킨다. 데이터의 최대 최소를 모를때 사용한다. 
    모든 Feature의 평균을 0, 분산을 1로 만든다. 이상치가 있다면 평균과 표준편차에 영향을 미치기 때문에
    데이터의 확산에 변화가 생긴다. 따라서, 이상치가 있다면 사용하지 않는 것을 권장한다.
    **(x - x의 평균값) / (x의 표준편차)**
    
2. Normalizer
    
    Standard Scaler, Robust Scaler, MinMax Scaler는 각 컬럼의 통계치를 이용한다면, Normalizer는 각 로우마다 정규화 된다. 각 변수의 값을 원점으로부터 1만큼 떨어져있는 범위로 변환한다. Normalize를 하게 되면 Spherical contour(구형 윤곽)을 갖게 되는데, 빠르게 변환할 수 있고, overfitting한 확률을 줄일수 있다. 
    
3. MinMaxScaler
    
    데이터의 값들을 0~1 사이의 값으로 변환시키는 것으로 최대값이 1이 되고 최솟값이 0이 된다. 각 변수가 정규 분포가 아니거나 표준 편차가 작을 때 효과적이다. 데이터가 2차원 셋일 경우, x축과 y축 값 모두 0과 1사이의 값을 가진다. 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다. 
    → Standard Scaler 처럼 이상치에 민감하다. 
    **(x - x의 최소값) / (x의 최대값 - x의 최소값)**
    
4. Robust Scaler
    
    모든 feature가 같은 값을 갖는다는게 Standard Scaler와 비슷하다. 하지만 평균과 분산 대신에 
    중앙값(median) 과 사분위수(IQR)를 사용하고 StandardScaler에 의한 표준화보다 동일한 값을 더 넓게 분포한다. 따라서 Standard Scaler에 비해 이상치에 영향이 적다. 
    → 이상치를 포함하는 데이터를 표준화하는 경우 효과적이다.
    

MaxAbs Scaler라는 것도 있는데, 최대 절대값과 0이 각 1, 0이 되도록 하여 양수 데이터로만 구성되게 스케일링 하는 기법이다.

<aside>
📌 IQR = Q3 - Q1 : 25% ~ 75% 타일의 값을 다룬다.

</aside>

<aside>
📌 Scaler를 통해 feature간의 크기를 맞추는 것은 중요하지만 모든 feature의 분포를 동일하게 만들
필요는 없다. 데이터가 한곳에 집중된 경우, 데이터를 표준화시키면 큰 차이를 만들어낼수 있기 때문에 데이터에 맞게 scaling할 필요가 있다.

</aside>

### fit_transform() vs transform()

fit()은 데이터를 학습시키는 메서드이고 transform()은 실제로 학습시킨 것을 적용하는 메서드이다.

fit_transform()은 fit()과 transform()을 한번에 처리할 수 있게 하는 메서드 train dataset에서만 사용된다. AI 모델은 train data에 있는 mean, variance를 학습한다. train data로 학습한 Scaler의 매개변수를 통해 test data의 feature가 Scaling된다. 

transform()은 train data로부터 학습된 mean 값과 variance 값을 test data에 적용하기 위해 transform() 메서드를 사용한다. 

### fit_transform()을 test data에 사용하지 않는 이유

fit_transform을 test data에도 적용하면, test data로부터 새로운 mean, variance를 학습하게 된다.
애초에 test data를 사용하는 이유가 새로운 데이터에도 적용이 되는지 확인하려고 테스트하는 건데 이걸 
학습해버리면 처음 보는 데이터에 대해 모델의 성능이 어느 정도인지 알수가 없다.

### 머신러닝 tree 모델에는 scaling이 필요하지 않은 이유

우선 트리 모델이란 데이터에 있는 규칙을 학습을 통해 자동으로 알아내서 트리 기반의 규칙을 만드는 것이다.
데이터를 적절한 분류 기준 값에 따라 몇개의 소집단으로 나누는 과정으로 데이터를 어떤 기준을 바탕으로 
분류 기준값을 정하는 지에 따라 알고리즘 성능이 크게 바뀐다. 

그런데 애초에 scaling하는 이유가 무엇인가? 데이터의 단위를 통일하기 위해서이다. 하지만 Tree 모델은
규칙을 기반으로 데이터를 분류한다. 따라서 각각의 데이터의 단위가 중요하지 않고 규칙에 따라 분류만 할수 있으면 된다. 따라서 scaling이 필요하지 않다. 

## 3) 데이터 학습하기

데이터를 학습시키는 방법은 다양하다. 앞서 사용한 fit_transform()을 사용할수도 있고, fit() 메서드를 
활용할수도 있다.

                

# 4. Ensemble(앙상블)

## 앙상블 학습이란?

앙상블 학습이란 여러 개의 분류기(classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 예측을
도출하는 기법이다. 

## 앙상블 학습의 특징

단일 모델의 단점을 여러 개의 모델들의 결합으로 보완했다. 성능이 떨어지는 서로 다른 유형의 모델들을
섞으면 성능이 향상되기도 한다. 예를 들어, random forest나 부스팅 알고리즘들은 결정 트리 알고리즘
기반인데 결정 트리는 쉽게 과적합한다는 단점이 있지만, 이를 앙상블로 보완할 수 있다.

## **앙상블의 유형**

1. Voting

2. Bagging - Random Forest

3. Boosting - AdaBoost, Gradient Boost, XGBoost(eXtra Gradient Boost), LightGBM(Light Gradient Boost)

4. Stacking

# 5. Model-driven, Data-driven

## 1) Model-driven AI(Model-centric AI)

기존의 model-centric AI는 고정된 데이터 셋에서 **코드**를 개선해서 더 나은 결과를 도출하는데 중점을 둔다.
인공지능 개발자들은 일반적으로 코드가 학습하는 데이터 셋을 실측 레이블들의 모음으로 간주하고, AI 모델은 레이블된 해당 훈련 데이터에 맞게 만들어진다. 따라서, 이러한 접근은 학습 데이터를 인공지능 개발 과정의 외부로 생각한다. 위에서 언급한 코드란 AI 모델 혹은 알고리즘을 뜻한다.

## 2) Data-driven AI(Data-centric AI)

반면에 data-centric AI는 코드는 독립적으로 생각하고, 데이터의 질을 높여서 결과물을 개선하는데 중점을 둔다. 즉, 데이터를 labeling하고, [증강하고](https://velog.io/@cha-suyeon/Data-augmentation), 관리하고, [큐레이팅](https://www.techopedia.com/whats-the-difference-between-model-driven-ai-and-data-driven-ai/7/34776)하는데 집중한다. Data-centric AI는 데이터의 전처리 과정이라고 생각할 수 있지만, 실제로는 데이터 수집, 모델 훈련, 오류 분석을 하는 AI의 주기를 
강조한다. 

## 결론

Model-centric AI에서는 많은 시간을 모델을 최적화하는데 사용하고 주어진 문제를 해결하기위해 가장 적절한, 최적의 모델을 찾는 것이 목적이다. 반면에, Data-centric AI에서는 데이터의 품질을 향상하는데 많은 시간을 할애하고 주어진 문제를 해결하기 위해 수집된 데이터에서 모순점을 찾는게 목적이다.

# REFERENCE

1. [parameter&return](https://blog.naver.com/PostView.naver?blogId=siniphia&logNo=221396370872)
2. [parameter&return2](https://teddylee777.github.io/scikit-learn/train-test-split/)
3. [Scaler_1](https://ebbnflow.tistory.com/137)
4. [Scaler_2](https://mingtory.tistory.com/140)
5. [fit_transform() vs transform()_1](https://deepinsight.tistory.com/165)
6. [fit_transform() vs transform()_2](https://for-my-wealthy-life.tistory.com/18)
7. [앙상블_1](https://libertegrace.tistory.com/entry/Classification-2-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5Ensemble-Learning-Voting%EA%B3%BC-Bagging)


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