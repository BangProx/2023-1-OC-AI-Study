# 5주차 : ML 기초

# 부제 : sklearn 활용하기

이번 주차에서는 코드 그 자체보다는 개념의 흐름에 집중을 해야한다. 결국 나중에 작성하는 딥러닝 코드들도 결국 이런 흐름을 코드로 표현한 것에 불과하다. 

[ML 실습](https://www.notion.so/ML-fa7fe0cfbd6a4ed49474cba2951b9408)

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