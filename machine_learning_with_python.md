Machine Learning

> 지도학습과 비지도학습을 구분하고, 지도학습은 분류와 회귀로 나뉜다
>
> 머신러닝적인 통계 접근과 통계학적인 통계 접근은 다르다!



## (교재) Introduction to Machine Learning with Python

> 원리에 대한 설명이 적은 편이므로 보충자료로 설명할 것
> ### CHAPTER2 지도학습
>
> **중점**: 분류와 회귀
>
> - 선형모델 - 결국 신경망 모델로 연결된다(딥러닝 학습은 모두 지도학습이다)
>
> - 결정트리
>
> 
>
> ###  CHAPTER3 비지도 학습
>
> GAN 에서 다룰 예정(GAN을 다룰 것인가...?)
>
> - 군집모델 
> - 전처리(Pre-processing)
>
> 
>
> ### CHAPTER4 데이터 표현과 특성공학
>
> 특성공학(Feature Engineering)
>
> - 원핫인코딩
> - 스케일링
>
> 
>
> ### CHAPTER5 모델 평가와 성능 향상
>
> 수업 앞쪽에서 보게 될 것
>
> 
>
> ### CHAPTER6, CHAPTER8 다루지 않을 가능성 큼
>
> 알고리즘 체인과 파이프라인 (진행하지 않을 수도 있음)
>
> - 그리드서치
> - 하이퍼 파라미터 튜닝
>
> ### CHAPTER7 - 자연어처리 (딥러닝에서 볼 것)








# 'Machine Learning'의 큰 틀

>## Agenda
>
>1. Artificial Intelligence: AI는 무엇인가
>
>- 인공지능을 만드는 것의 의미
>
>
>
>2. Machine Learning(Gradient Descent): 학습 원리
>  - 학습을 시키는 "머신"이 무엇인가?
>  - **Gradient Descent**: 경사하강법
>
>
>
>3. Model Validation
>  - 모델 검증: 모델이 실제로 사용하기에 적합한지 평가
>  - 잘 될것인지를 확인하여 "잘 될건지"(미래) 파악하는 것
>
>
>
>머신러닝으로 할 수 있는 것은 기본적으로 두가지 (지도학습 - 분류와 회귀): 숫자나 범주를 예측하는 것
>
>4. Regression Analysis: 연속적인 숫자를 예측하는 것
>5. Logistic Regression: 분류(범주, category). 범주예측
>
>
>
>결정트리
>
>6. Decision Tree
>7. Random Forest(Ensemble)
>
>
>
>비지도학습 알고리즘
>
>8. K-means Clustering
>9. Association Rules





## 1. Artificial Intelligence

> - 우리는 지능을 만들고 싶다
> - 역사적인 배경
> - 사람이 만든 "지능"



지능을 한가지로 정의할 수 있느냐?

지능을 만든다면, 지능이 무엇인지 명확히 정의할 수 있어야 한다





### 1) Definition

- 인공지능이란 인공장치(=기계)들의 **지능을 설계**하는 것(MaCarthy, 1956)

  - 인공장치: 사람이 만든 기계 (컴퓨터가 아닐 수 있음)
  - 인공장치가 인간의 지능을 모방하는 것

  - 물리적인 행동이 아니라, 논리적인 행동(지능)을 모방하는 것

  - 생각하고 판단하는 게 아니라, 사람이 해야하는 일을 대신해주는 것

    예) 계산기 

  

  인공지능이 엄청난 것이라고 생각하지 말자

  사람이 가진 지능의 아주 작은 부분을 구현하는 것

  

- 지능(Intelligence): 인간이 행하는 지적 작업의 주체
  - 지적작업: 근육이 아니라 인간의 두뇌활동에 의해 이루어지는 작업
  - 생명체가 생존 환경의 변화에 적응하기 위해 인지적 기능을 변화시키는 능력



### 2) History

#### (1) 1943년(여명기): 사람의 두뇌가 동작하는 것과 비슷한 기계를 만들 수 없을까?

- 두뇌 논리회로 모델링(McCulloch & Pitts)
- 1940년대 컨셉은 나왔지만, 실체화시킬 기술이 없어 이제서야 Deep learning으로 이어짐



#### (2) 1956년(태동기)

- 다트머스 회의에서 AI 용어 탄생



#### (3) 1956년 ~ 1970년(1차 인공지능 붐)

- 수동적 대화시스템

- 지능을 "기호처리"로 정의함

  - 언어는 기호이고, 기호의 교환을 통해 의사소통(communication)이 가능해질 것이라고 생각

    예) 챗봇

- pc도 없던 시절...



- 1971년~1979년(1차 빙하기): 프레임(고려해야할 범위)의 문제

  - 우리가 대화를 하는 데 있어서 생각보다 고려해야 할 것이 너무 많구나!

    예) '아침 드셨어요?'

    ​			사람은 '오늘 아침'으로 이해 가능(언어지능이 발달했으므로)

    ​			기계는 명확하게 설명해주어야 함

  - 이유: 컴퓨터 기술의 한계, 너무 어려운 주제...



- 1976, 최초의 pc 등장과 컴퓨터의 보급화
  - 지능을 다시 정의해보자
  - 기호의 교환이 아니라, 지능은 지식(knowledge)이다



#### (4) 1980년~1995년(2차 인공지능 붐)

- 전문가시스템(expert system) 활용(지능은 지식)

  예) 의사의 지식을 컴퓨터에 전달하고, 필요할 때 전문가에게 물어보는 것처럼 컴퓨터에 물어보자



- 1998년~2009(2차 빙하기): 지식획득 병목의 문제

  - 우리의 지식은 20%의 '형식지'(표현해낼 수 있는 지식)와  80%의 암묵지(남에게 정량적으로 명확하게 표현할 수 없는 지식)로 이루어져 있다

  - 80%의 암묵지를 컴퓨터에 제대로 전달할 수 없으며, 20%의 형식지만으로는 제대로 된 의사결정을 할 수 없다

  - 수집해서 정량화시켜 전달시키는 것이 어려움

    

#### (5) 2010년~ 현재(3차 인공지능 붐): 빅데이터와 딥러닝(지능은 학습)

- 사람의 지능은 어디에서 왔을까?

  예) 사람이 태어나 빈 공간에 혼자 두고 키운다면, 말을 할 수 있을까?

- 우리는 어떻게 말을 할 수 있게 되었을까?

  - 지속적으로 데이터(엄마, 아빠)가 들어오고, 패턴을 찾아내어 성대로 내보내게 됨

- 왜 그전에는 지능을 학습이라고 생각하지 못했나?

  - 컴퓨터가 학습하려는 디지털 데이터가 없음
  - 1970년대 pc 보급으로 디지털데이터 축적, 인터넷 연결과 스마트폰으로 데이터가 폭발적으로 증가하며 컴퓨터를 학습시킬 충분한 디지털데이터를 확보할 수 있게 됨

- 데이터를 주면 컴퓨터가 알아서 배울 수 있게 되겠구나!

- 그래서 "학습"(learning)이 붙게 됨: 디지털 데이터를 집어넣어서 특징을 찾게 하는 것이구나!

- 정형/비정형이든 컴퓨터에게 디지털데이터를 제공할 수 있는 이 시대에 자동으로 특징을 집어넣을 수 있게 됨

- 지능을 사람이 만들어서 집어넣는 것이 아니라, 컴퓨터가 데이터로 스스로 배울 수 있게 하려는 시대



---

우리는 지금 3차 인공지능 붐의 시대에 살고있다

---





### 3) AI Type

#### (1) 약한 인공지능(A**N**I: Artificial **Narrow** Intelligence)

- Narrow: 많은 양의 데이터를 처리하여 **특정 기능**만 수행
- "특화 지능": 특정기능에서는 사람보다 잘하지만, 모든 지능을 갖지는 못함
- 데이터를 가지고 기계가 특정 지능을 학습하도록 하는 것이 지금의 "약한 인공지능 시대"



#### (2) 강한 인공지능(A**G**I: Aritifical **General** Intelligence)

- 사람처럼 생각하고 판단하는 **범용** 인공지능
- 아직까지 존재하지 않는다
- 인간의 Natural Intelligence(자연지능)와 흡사



- 특이점(Singularity): 인간과 인공지능 사이의 임계점

  - 레이 커즈와일: 2045년에 특이점이 온다고 예측

  

#### (3) 슈퍼 인공지능(ASI: Artificial Super Intelligence)

- 특이점을 지나 인간의 지능을 넘어선 인공지능 - Transcendence(초월성)



기계가 인간을 공격할 것인가?

기계에게 가장 훈련시키기 어려운 것이 욕망

기계가 인간을 공격하는 이유는 1) 버그 2) 나쁜사람이 나쁜 용도로 사용해서



기술적인 부분도 중요하지만, **인공지능 윤리**가 대두됨





### 4) Turing Test

- Alan Turing, 1950
- 컴퓨터가 지능을 가지고 있는지 여부를 조사
- 우리가 기대하는 만큼의 성능을 내는지 평가하는 것으로 의미가 변해가고 있다
- 질문자가 **인간과 컴퓨터에게 같은 질문**을 하여 **인간의 답**과 **컴퓨터의 답**을 **구분할 수 없으면** 컴퓨터가 지능을 갖고 있는 것으로 볼 수 있음 
- CAPTCHA(Completely Automated Public Turing test to tell Computers and Humans Apart): 사람인지 기계인지를 구분하는 장치





현재 컴퓨팅의 시스템을 "폰 노이만"이라고 한다



폰 노이만(맨하탄 프로젝트): 앨런 튜링의 지도교수

- 야사에서는 폰 노이만이 앨런 튜링의 아이디어를 훔쳐 컴퓨팅 시스템을 만든 게 아닌가 하는 썰이 있다



앨런 튜링: 게이, 당시 영국은 동성애가 불법. 화학치료를 받다 사과에 독을 묻혀서 먹고 자살



스티브 잡스: 앨런 튜링에게 바치는 로고가 아닌가? 하는 야사가 있음

- 깨물어먹은 사과. 무지개빛





### 5) Best Practices & Issues

#### (1) Technologies

- 딥러닝
  - 사물인식(object detecting): 명사만 알면 됨
  - 사물인식 및 이미지 보정
  - 시각장애인 주변 설명(image captioning): 동사, 형용사까지 알아야 됨
  - AI Reconstruct Photos(GAN 생성모델)
  - Deep Fake(GAN)
- 강화학습
  - OpenAI Solving Rubik's Cube(강화학습)

* 머신러닝은 "통계분석"에 가깝다

- Amazon GO
- ZOZOSUIT
  - 신체사이즈를 측정(시각인식), 사이즈가 맞는 옷을 추천

- etc
  - 투자분석 리포트: 애널리스트 15명이 한달 간 해야할 분석작업을 5분만에 처리
  - IBM Watson: 한해동안 발표되는 암 관련 논문 44,000개
  - 영화추천: 영촤 출연 배우 중 고객이 선호하는 배우 이미지로 광고 노출
  - 콘텐츠 생성: Google AutoDraw



#### (2) Issues

- Amazon Alexa: 아마존 인형의집 사건
- NIA Special Report
  - 인공지능 악용에 따른 위협과 대응 방안
  - 적대적 스티커(Adversarial Patch)



#### (3) Industrial Applications

- 제조업
- 자동차
- 소매업
- 금융업
- 운송업
- 헬스케어
- 엔터테인먼트: 맞춤형 개인화 서비스





## 2. Machine Learning (Gradient Descent)

> AI를 만드는 두가지 방식: ML/DL
>
> Intelligence(사람의 지적인 작업)
>
> 예측(Predict) : 과거의 데이터의 패턴이 미래에도 나타날 것이다
>
> - Regression(회귀): 수치를 예측
> - Classification(분류): 범주를 예측



### 1) ML에 대한 일반적인 설명



#### What is learning?

- 행동의 변화
- 학습 전 / 학습 후



- 학습이란 "어떤 **1) 작업**에 대해 **2)특정 기준으로 측정한 성능**이 **3)새로운 경험**으로 인해 향상되었다면, 그 프로그램은 어떤 작업에 의해 **4)향상**되었다면, 그 모델은 어떤 작업에 대해 특정 기준의 관점에서 새로운 경험으로부터 '배웠다'라고 말할 수 있다." <Tom M.Mitchell, 1998>

  1) 작업: 예측/분류

  2) 특정 기준으로 측정한 성능: 정량적

  ​	측정

  - 정성적 예) 크다, 많다, 좋다, 빠르다

  - 정량적 예) 2개 틀림, 30보다 작음

  3) 새로운 경험: parameter update (gradient descent에 의해)

  4) 향상: 측정한 성능에 긍정적인 영향을 줌 (틀린 것이 줄고 맞은 것이 늘었다)



- 학습 이후 새로운 데이터(test data)에 대하여 학습된 내용(train data)으로 처리하는 것



#### 우리는 왜 학습하는가? 

- "긍정적으로" 변화하려고 (학습 전 후 행동의 변화, 사고의 변화)

- 변화(Change)는 좋을수도, 나쁠 수도 있음 (=Risk)
  - Positive
  - Negative



#### Machine Learning에서 "Machine" 은 무엇인가?

- 컴퓨터가 아니다.
- Computer는 H/W. 컴퓨터"로" 학습하는 것(컴퓨터는 도구)



예) 파란색, 빨간색 집단 분류

​	그래프에 그리면 좌표로 표현 가능하고, 일차방정식(y = ax + b)으로 구분선을 그리는 것이 가능해진다



- Machine을 **수학의 함수f(x) (y = ax+b)**로 생각하자

- 무엇을 학습시키는가? a와 b (=모델에서는 a와 b를 Parameter라고 한다)

- a와 b를 변화(change)시키는 것

  = Parameters인 a와 b를 주어진 데이터에 최적화(Optimization) 시킨다

- 학습이 된 함수를 "모델"이라고 한다





#### Machine Learning에서 "Learning(학습)" 은 무엇인가?

- 변경의 대상: a와 b
- 제공된 데이터: x와 y
- 파라미터의 업데이트





---

#### 따라서 Machine Learning이란,

- 사람이 해야 할 의사결정(예측)을 학습된 함수(모델)을 통해 인공지능이 대신하게 하는 것
- 정해진 함수를 주어진 데이터를 가지고 학습한다

---



#### Machine Learning Definition

- **1) 머신**이 **2) 코드**로 명시되지 않은 동작을 데이터로부터 학습하여 실행할 수 있도록 하는 알고리즘

  ​	1) 머신: function  2) 코드: application/program 

  - 데이터로부터 일관된 패턴 또는 새로운 지식을 찾아내 학습하는 방법
  - 학습된 알고리즘(model)을 적용하여 정해진 업무를 처리

- 현실세계의 다양한 문제를 수작업에 의한 프로그래밍으로 대응하기 어려움

  - 개발자가 만든 것 이상을 수행할 수 있다는 의미

- 학습할 수 있는 것과 학습할 수 없는 것을 구분하는 것이 중요



#### Machine Learning Example

- 사람(Developer)이 만든 알고리즘(Software) vs. 기계(Machine)가 학습한 알고리즘(Model)
- Spam e-mail vs. Ham e-mail
- 레이블이 반드시 필요 (레이블을 토대로 지도학습) - 그래야 검증이 가능해짐

- 이렇게 레이블을 붙여주는 것이 "데이터 댐" 사업





#### Machine Learning 구조

Machine Learning

​	**Supervised** Learning (교재 chapter 2)

​		: Develop predictive model based on both input(=x) and output(=y, label) data

​		Regression

​		Classification

​			이진분류(2개)

​			다중분류(3개 이상)

​	**Unsupervised** Learning: Group and interpret data based on only input data(=x)

​		Clustering(군집, 연관)



#### Supervised Learning(지도학습)

> 우리가 배우는 machine learning은 다 지도학습이라고 생각하자!



- 데이터에 존재하는 **특징(Feature)**을 바탕으로 처리 (수치예측, 범주예측)
  - feature: 통계학의 변수(Variable)
- input(x)에 대한 output(y)을 제공: data(labeling)가 제공되어야 학습할 수 있다
- input(x) Data와 output(y) Data(label)의 **관계**를 학습





#### Algorithm

| 지도학습                           | 비지도학습                      |
| ---------------------------------- | ------------------------------- |
| 회귀분석(Regression Analysis)      | 주성분분석(PCA)                 |
| 로지스틱 회귀(Logistic Regression) | K-평균 군집(K-means Clustering) |
| 의사결정 나무(Decision Tree)       | 연관 규칙                       |
| 랜덤 포레스트(Random Forest)       |                                 |
| 신경망(Neural Network)             |                                 |



#### Relationship

>  AI > ML > DL

- ML: 정형(RDB) 데이터인 경우가 많음

- DL: 비정형(사진, 음성, 소설, sns 데이터)

---

쓰임새가 다르다고 생각하면 된다. 그러나 DL도 ML안에 포함된다.

---





### 2) ML에 대한 기술적인 설명 - Gradient Descent

> Gradient: 경사/기울기 Descent: 하강
>
> 우리가 하는 모든 학습의 원리
>
> - 학습: Parameter Update
> - parameter를 update하는 원리가 바로 "gradient descent"이다
>
> Optimization Method



#### Function

- y = wx + b
  - w: weight(가중치)
  - b: bias(편향)



#### Loss Function() for Regression Analysis

- 오차 함수, Loss Function(): **1) 실제값**과 **2) 예측값**의 차이(**3) 오차**)를 비교하는 지표

  1) 실제값: y

  2) 예측값: y_hat(모델에 의해 나온 y값)

  3) 오차: 실제값과 예측값 사이의 차이(= error/loss/cost)

  - 점과 선 사이의 거리 

  - 오차를 계산하는 수식을 수학적으로 표현하면: y - y_hat
  - loss function은 y에서 y_hat을 뺀 값들의 합, 모두 양수로 만들기 위해 제곱함: L(y, y_hat) = (y - y_hat)**2
  - 제곱한 값이 작을수록 좋은 예측모델



​		예) Model : y_hat = 2x + 1 vs. y_hat = 3x + 0

| y    | x    | y_hat           | Loss              | y_hat2        | Loss              |
| ---- | ---- | --------------- | ----------------- | ------------- | ----------------- |
| 3    | 0.9  | 2*0.9 + 1 = 2.8 | (3-2.8)**2 = 0.04 | 3*0.9+0 = 2.7 | (3-2.7)**2 = 0.09 |
| 9    | 3    | 7               | 4                 | 9             | 0                 |
| 15   | 4.8  | 10.6            | 19.36             | 14.4          | 0.36              |
| 21   | 7.2  | 15.4            | 31.36             | 21.6          | 0.36              |



- 모델 값을 바꿔본다 (파라미터 값을 바꿈 -- 학습)
- 첫번째 모델보다 두번째 모델의 전체 오차의 합이 적다 -- 더 나은 모델이라고 할 수 있다
  - 합은 너무 크므로 보통 평균을 MSE(Mean Squared Error)를 작게 만드는 방향으로 한다
- w와 b를 크게하거나, 작게하면서 오차를 줄여나간다



- b의 값에 따라서 MSE이 작아질수도, 커질수도 있다
- w는 고정되어 있고, b만 바뀐다고 생각해보자

  - MSE = (y-y_hat)**2 = (y-(wx+b))\*\*2 = (y-wx-b)\*\*2
  - MSE 안에도 사실상 b가 들어가있고, 그래프는 사실상 b에 대한 2차방정식이다
  - 우리는 MSE 최소지점의 b를 찾고싶은 것
  - 그것이 b를 업데이트하는 학습의 목표



- 오차함수의 그래프를 알고있다고 한다면 b를 줄여야하는지 늘려야하는지 알 수 있겠지만, 실제로는 그래프를 모른 채로 오차를 줄여나가야한다
- 그러므로 b에(2차함수에) 미분을 한다
  - 미분이란? 순간변화량(delta) (=접선의 기울기)
  - 2차함수를 미분하면 1차함수가 됨
  - b지점에서 미분을 하면 접선이 생기고, 접선의 기울기를 계산할 수 있게 됨



- 편미분을 통한 dw, db 계산과정
  - MSE = (y-y_hat)**2 = (y-(wx+b))\*\*2 = (y-wx-b)\*\*2
  - MSE를 w에 대해 편미분(dw)
    - 1단계: 바깥을 미분   2(y - wx - b) 
    - 2단계: 안쪽 미분  (y - y_hat) * 2 * -x
  - MSE를 d에 대해 편미분(db)
    - 2 * (y -wx - b)
    - 2 * (y - wx - b) * (-1)
    - (y - y_hat) * 2 * (-1)



- 경사하강법: b 값에서 기울기값(db)을 빼서 원래 있던 b값에 다시 할당

  - b - 기울기(Gradient)

  

- 최소값에 도달하면 기울기가 0이 된다

- db가 0이 되므로 b에 b를 계속 넣기만 할뿐, 더이상 학습이 되지 않는다

- 기울기가 0인 지점을 찾으면 된다



---

"b 값에서 기울기(Gradient)만큼 빼주면, error가 줄어들(Descent) 것이야!"

#### 경사하강법:

어느 지점에서든 미분의 접선의 기울기값을 빼주면(+ 이든 - 이든) 오차가 적어지는 방향으로 자동으로 이동할 것이다 (자동으로 업데이트 할 수 있을 것이다)

---



#### 감마(Learning Rate, Step Size)

- 그러나 기울기값이 무척 커서 한번에 너무 많이 이동하게 된다면, 방향이 맞더라도 에러가 커질 수도 있다(부호가 의미하는 것은 방향이지 크기가 아니므로)

- 기울기 값을 작게 제어해주는 매커니즘이 필요 (감마, 0~1 사이 값)
- 기울기 값에 감마 값을 곱해준다. 천천히 학습하도록
- 감마: Learning Rate(학습율) or step size 라고 부르게 된다



#### MSE와 w, b

- b뿐만 아니라 w도 같이 움직여야 한다 (따라서 GD는 곡면에서 동작한다)
  - 공간상에서 최소값을 찾아가는 방향으로 동작하게 된다

- 제곱오차의 면적을 최소화시키는 방향으로 모델을 만들어간다(w와 b의 값을 update)





#### Global Minimum vs. Local Minimum

- convex nonconvex

- 학습의 목적은 global minimum 을 찾아내는 것
- GD는 local minimum 에 빠진다는 문제점이 있다
- 해결하는 방법
  - 여러군데에서 해본다 등등... (딥러닝에서 배울 것 - optiminzation method)
- convex 가 아니라 nonconvex 에서 보통 문제가 나타나는 것을 보게 될 것





---

#### Gradient Descent 정리

- 데이터 기반 학습에는 gradient descent라는 개념이 무조건 적용된다

- Gradient Descent 의 식
  - w = w - r * dw(delta w)
  - b = b - r * db(delta b)

- gradient descent를 통해 주어진 데이터에 최적화된 함수를 자동으로 만들어낼 수 있다

- 원래있던 값에서 경사값(Gradient, 미분값)을 빼주면 오차가 감소(Descent)할 것이다

- 너무 많이 이동하면 안되니까, 0~1 사이의 값 r(감마)을 곱해준다 -- hyper parameter 라고 부름

  (이동거리를 비율로 줄여준다는 의미)



- w, b: 학습의 대상인 "파라미터 (학습에 의해 자동으로 업데이트 됨)
- r: 뭐가 제일 좋은지 해봐야 안다(찾아야 한다). 실행해봐야 아는 값 "하이퍼 파라미터"
  - 여러가지를 넣으며 최적의 값을 찾는 과정이 "하이퍼 파라미터 튜닝"

---

하이퍼 파라미터 튜닝: 랜덤 포레스트에서 언급되고, 딥러닝에서 매우 중요해진다

하이퍼 파라미터는 답이 없다. 해봐야 안다







___

### 총정리

- 우리는 인공지능을 만들고 싶다
- 그것을 위해 machine이라는 '함수'를 학습시키는데
- x, y는 데이터로 주어져야 하고, w,b를 데이터에 최적화하여 학습시키는 것이다
- 데이터를 지속적으로 제공해준다면
- 데이터의 특징을 확인하게 될 것이고, (Data Analytics)
- 특징: 모여있는 모양, 떨어져있는 모양 두가지 (통계의 "분포")
- 학습을 통해 새로운 형태의 처리방법(모델)을 찾아내게 될 것이고
- 학습을 시키기 위해 중요한 것은 정답(y)과 기준(작업에 따라 달라짐)이다. 
  - 수치예측(Regression)에서는 그 기준이 MSE
- 오차가 줄어드는 방향으로 자동으로 업데이트하기 위해 Gradient Descent를 사용한다.
  - 원래 있던 값에서 편미분 값을 학습률을 곱해서 빼주는 것
- 이것을 반복하면 최적의 함수를 찾아낼 수 있게 되고, 최적의 함수를 가지고
- 미래 어떤 결과가 나올지 예측할 수 있다



- 사람이 하던 일을 기계(함수)를 이용해 자동으로 하고싶다

___



파라미터의 초기값은 0~1 사이로 보통 랜덤하게 세팅된다



학습을 많이 시킬수록 오차는 점점 줄어들 것

하지만 그것이 정말 좋은 모델일까?





## 3. 모델 검증(Model Validation)



- 함수를 학습시켜 모델로 만들면,
- 모델로 예측작업이 가능해진다
- 모델이 예측을 한다면, 그 결과를 믿을 수 있어야 한다

- Validation이란? 사용에 적합한가 확인하는 과정





### Enough vs. Not enough



1차방정식으로 만든 함수는

우리가 만든 모델이 데이터를 설명하기에 충분하지 않다



### First-Order(1차함수) Model vs. High-Order(다항함수) Model

(교재 chapter4. p.293)

2차함수: y2 = w1x**2 + w2x + b

5차함수: y5 = w1x^5 + w2x^4 + w3x^3 + w4x^2 + w5x + b



파라미터의 수가 늘어나게 됨

모델의 데이터에 대한 설명능력: Model Capacity

Model Capacity에 직접적으로 영향을 주는 것 중 하나가 파라미터의 개수





모델을 사용해서 데이터의 분포를 설명/표현하려고 함

모델의 w,b가 데이터를 설명하는 것

설명할 수 있는 애들(w,b)이 많아지면 설명할 수 있는 것도도 늘어나게 됨

일반적으로 파라미터 개수가 늘어나면 capacity가 좋아진다



### Model Capacity

>  Capacity: 모델의 데이터에 대한 설명력



#### 1) Parameter의 개수



#### Data 산점도

전기 사용량과 바닥면적의 상관관계



- 1차원 모델을 만들면?
- 2차원 모델을 만들면?
- 9차원 모델을 만들면?



모델링을 할 때, 한가지를 만드는 게 아니라

여러 모델을 만들어서 모델 별로 오차가 작은지를 비교해봐야 한다

모델 중 가장 적합한 모델을 찾아야 한다

한번에 좋은 모델이 나오는 게 아니라, 다양한 시도를 통해 가장 좋은 모델을 찾아낸다



정확히 MSE를 계산해서 좋은 모델을 판단해야 한다





딥러닝으로 갈수록 파라미터의 개수가 늘어나게 되고, 

특히 자연어처리 모델들(1천 8백억개...)

복잡한 





### Training Error



Machine Learning에서 Learning이란?

경사하강법을 사용해서 모델의 파라미터를 최적화시키는 것



이것이 정말 Learning인가, Training에 가깝게 느껴지지 않는가?

MSE 을 Training error 라고 하기도 한다



- 기본적으로 학습은 training error를 최소화하는 방향으로 진행
- Training Error
  - Training data에 model을 적용하여 확인한 실제값과 예측값의 차이(오차)
  - mean((y - y_hat)**2)
- 문제점) 여러 개의 Model을 생성 후 각각의 Training Error 를 비교
  - 실무 데이터에 적용하기 위한 최적의 Model을 선택
  - Model생성(model.fit(x, y))과 평가(model.predict(x))에 **같은 데이터를 사용**하여 **부작용** 발생



- 오차(training error)를 계산하기 위해서는

- sklearn이 필요





1차 모델은 계산이 간단(colab 참조)

5차 모델:  y5 = w1x^5 + w2x^4 + w3x^3 + w4x^2 + w5x + b

x, y 는 있지만 x^2 ~ x^5는 없음. 만들어줘야 함

다차원속성을 만들어줘야 한다

- sklearn 의 PolynomialFeatures 를 통해 만들 수 있다





MSE는 0이 될 수 없다

왜? 데이터에 오차가 0이 될 수 있는가?

예) 데이터가 5000개... 이론상 5천차함수를 만들면 오차가 0은될 수 있을 것이다

그러나 그것이 의미는 있으며, 현실적으로 훨씬 많은 데이터를 다루는데 과연 0으로 만드는 것이 가능한가?

따라서 MSE(오차, 잔차)가 0이되는 것은 사실상 없다





 

9 차모델을 쓰는 게 타당할까?

타당하지 않다. 왜?





Data Analytics

과거의 특징을 보는 것



우리가 하는 머신러닝도 과거의 데이터를 사용하여 학습

우리가 만든 머신러닝의 모델은 과거 행동의 특징을 파라미터를 통해 학습하는 것





데이터 분석의 목적

1. 과거 행동의 특징을 확인 - EDA
2. 과거 행동의 특징을 학습해서 미래 행동에 영향을 줌, 미래 행동의 결과를 예측(predictive model)



우리의 모델은 과거 행동에 최적화되어 있을 뿐, 미래 행동도 예측할 수 있을까?



Training Error의 문제점

Model생성(model.fit(x, y))과 평가(model.predict(x))에 같은 데이터를 사용하여 부작용 발생



과거 데이터에 최적화되어 있어서, 미래에도 그 정도의 MSE가 나올 수 있는가?



이러한 부작용을 "과적합(Overfitting)" 이라고 한다

- 과거데이터에"만" 엄청나게 잘맞다는 뜻.
- 미래 데이터에 잘 맞느냐? 모르겠다.





### Overfitting

- 학습한 결과(model)가 **Training Data에만 최적화된 모델**을 생성
  - 학습에 사용되지 않은(미래 발생) 데이터에서 성능이 급격하게 낮아짐(오차가 급격하게 올라감)
- 부적합(Under Fit) vs. 최적합(Ideal Fit) vs. 과적합(Over Fit)



- 사용하기에 적합하지 않다(not validated)





- training error만 보는 것은 타당하지 않다
- 왜? 학습한 결과로 평가를 하기 때문에 신뢰성이 없다
- 따라서 dataset을 두개로 쪼개어 하나로 모델을 만들어서, 모델을 만들 때 쓰지 않은 데이터로 testing error를 본다
- testing error 가 낮은 것이 미래 행동도 잘 예측할 것이라고 보는 것이 타당하다





### Testing Error

- 과거 데이터를 두개로 쪼개서 하나는 training(fit), 하나는 testing(predict)에 사용
- 모델을 **학습(모델생성)** 후 반드시 **평가(모델평가)**가 필요
- Testing Error를 활용하여 **판단의 객관성** 측면에서 학습(생성)된 모델을 비교
  - Training Data: 학습(모델생성)을 위해 제공되는 데이터
  - Testing Data: 학습결과를 평가(모델평가)하기 위한 데이터
- 데이터의 크기에 따라서 일반적으로 8:2 또는 7:3 등의 비율로 구성

(Chapter2 p.73 선형모델, p.76)



- sklearn.model_selection train_test_split 사용

- train data 와 test data가 오차가 둘다 줄어들어야 함



- 하나의 덩어리인 데이터를 두개로 쪼개자

- 어떤 비율로 쪼개야 제일 좋을까? 해봐야 안다.
- 쪼개는 비율도 "하이퍼 파라미터"



- 데이터프레임 그대로 쪼개면?
- 9개 컬럼이 그대로 옴 (614, 9) (154, 9)
- 실제로는 두개만 씀



- X, y가 구조상으로는 시리즈이지만 array로 쪼갬
- X_train, X_test (614, 1) (154, 1)
- y_train, y_test (614, 1) (154, 1)



- training error vs. testing error

  평가는 testing error를 기준으로 model selection 한다



### Training Data vs. Testing Data

#### 과정

1. Data split (train : test)

   1) Training Data

   ​	-> Machine Learning Algorithm(예. Regression): Model_1.fit(X, y)

   ​	-> Models: 여러 모델을 만듦(1차, 5차, 9차 등)

   2) Testing Data

   ​	-> Models: .predict(X_test) --> y_hat --> Metric(예. MSE)

   ​	-> Model Select



training data로 모델을 만들어 testing data로 예측값을 생성하고 모델의 성능을 평가

최적의 모델을 선택



- Model을 select 하는 과정에서 test data를 봐버리면,







#### Machine Learning / Deep Learning modeling의 궁극의 목적

- 일반화된 모델을 만드는(학습시키는) 것

- 일반화란? 모델 생성(학습) 시 사용되지 않은 데이터에서도 유사한 성능을 제공

  (Training Error, Test Error(=MSE)의 갭의 크면 안된다)

- overfitting되지 않은 model을 만드는 것이 우리의 목적



- 이 모델을 사용해서 실제로 만나게 될 오차가 MSE와 가까울까?
- training error보다는 test error가 실제보다 더 가까울 것
- 그러나 사실 model selection에도 학습은 일어난다
- model select 까지 학습과정으로 본다면, 사실 test data도 학습에 쓰이는 것



여러개 모델 중 하나를 고르는 것도 학습이다



### Validation Approach

> 생각해보면 좋을 문제
>
> validation approach는 왜 할까? 고민해보자
>
> - generalization error를 추정하기 위해 사용
> - 미래에 만날 에러에 좀 더 가까운 에러를 추정해보기 위해서



- Testing Error
  - Testing Data에 **최적 Model**을 적용해 얻은 실제값과 예측값의 차이(오차)
  - Testing Error를 사용하여 실무 데이터에 대한 **Generalization Error**를 **추정**
- Testing Data역시 모델평가 과정에서 사용되어지는 문제점 발생
  - Training Data를 Training Data와 Validation Data로 분리하여 모델을 평가





따라서 데이터셋을 3개로 쪼갠다

Training Data: Model.fit

Validation Data: Model.predict, Model-select -- 모델을 1개 고름 // 학습은 여기서 끝남



최종적으로 select된 모델을 가지고, generalized error를 추정

Testing Data: 미래의 오차가 어느정도 되는지 추정하기 위해서 사용



- 사실상 train, test로 model select 해도 크게 상관은 없다

  (실무에서는 validation을 하게 될것)



- 문제: sklearn에는 train_test_split 밖에 없다
  - 방법: train_test_split을 두번한다
  - 데이터를 두개로 쪼갠 후(Test 생성), 하나를 한번 더 쪼갠다(Train, Validation)





### Generalization Error 일반화된 모델

- 학습에 사용되지 않은 데이터에서도 유사한 오차가 나오기를 바람





교차검증

cross validation



미래에 발생할 오차를 최소화하기 위해 최선을 다할 뿐





---

### 오늘내용 정리

- 모델은 하나만 만들지 않는다
- 여러개를 만들어서 사용하기 적합만 모델을 select한다
- 예측모델에 대해서만 보았고, 실제 분류모델을 검증하는 방법을 달라지게 될 것이다

- model capacity가 달라지는 여러가지 이유가 있다
  - 다항함수로 order를 올리는 것: parameter개수가 많아지면 capacity가 올라간다
- 시각화 뿐만 아니라 error수치를 보아야 하는데, error수치는
- training error - 객관적으로 신뢰할 수 없기 때문에
- test error 가 좀 더 객관적이고 타당할 것이라고 본다
- 여러 모델의 mse를 비교해서 가장 mse가 낮은 모델을 선택하는 것이 타당하다
- 그러나 전체 과정에서 model select 도 학습과정에 포함된다고 보기 때문에
- 데이터셋을 3개로 나누어 (train, validation, test)
- train 으로 모델링, validation으로 model select (학습은 여기서 끝난다)
- test로 generalization error를 추측
- test 데이터는 keep해뒀다가 최종 모델로 미래 오차를 예측하는 데 쓰인다

---







## 4. Regression Analysis

> 수치 예측
>
> model: y=wx + b 일 때, y가 연속형 데이터
>
> 일반적으로 x도 연속형 데이터이다(아닌 경우도 있다)
>
> 분할하기 전 scaling, encoding을 해서 성능을 좋게 만들어준다



### Supervised Learning - Regression Analysis



- 과거의 결과값을 기준으로 미래의 결과값(수치)을 예측하는 방법

  - 과거 결과값: 지난 습도가 50일 때, 불량품 수량이 4.5
  - 미래 예측값: 이번달 습도가 65일 때, 불량품 수량을 예측

- 미래에 발생할 결과값이 "과거의 **평균**으로 돌아간다(회귀)"는 의미

  - "평균"으로 돌아간다는 의미에 대해서는 통계학에서 더 자세히 알아본다

- 회귀모델: " y ~ wx + b"를 사용하여 w와 b의 값을 추정

  - ~: "틸드"로 읽는다
  - 등호(=)는 좌변과 우변이 같다는 의미
  - 모델의 입장에서는 같겠지만, 정말 정확하게 같은가? 아니다
  - 통계학에서는 ~(틸드) 를 사용하고, 우리가 만든 모델이 x를 통해서 y를 설명(표현)한다 는 의미

- y: Output(종속변수, 반응변수), x: Input(독립변수, 설명변수)

  - x의 값에 따라 y의 값이 결정된다는 의미

  - input 과 output의 관계를 설명한다

  - 머신러닝과 통계학 용어가 혼재되어 사용되므로, 같은 것이라는 걸 이해하면 된다

    

- 최소제곱법(Least Squared Method), 잔차제곱합(Resideual Sum of Squares), 평균제곱오차(Mean Squared Error) 



- ex) 습도에 따른 불량품수량 예측

  - 예측모델은, x값 변화에 따른 y값의 변화를 보려고 하는 것
  - 만약 w가 0이라면? x값이 어떤 것이 들어가도, y_hat 은 b밖에 없음. 모델을 만드는 것 자체가 의미가 없어짐
  - 통계에서의 회귀분석은 다양한 가정과 검증을 통해 이루어지게 되는데, 머신러닝에서는 모델에 대한 가정을 그만큼 철저하게 하지는 않지만 w가 0인지 아닌지를 검증하기 위해 "상관계수"를 사용한다

  

- 상관계수(Correlation Coefficient): x, y의 관계에서 어느정도 의미를 가지는지 평가하기 위한 계수

  - -1 <= r <= 1
  - 양의 상관관계(정비례): x가 늘어나면 y도 늘어남
  - 음의 상관관계(반비례): x가 줄어들면(늘어나면) y가 늘어남(줄어듦)
  - r = 0: 기울기가 0인 형태. x와 y는 관계가 없다 / 독립이다
  - 0에 가까운 값이라면 회귀모델을 만들어서 예측하는 것이 의미가 없을 수 있다

  

  상관계수를 사용하는 이유

  1. x와 y의 관계 확인
  2. x1과 x2의 관계를 확인
     - 주의) ***** 혼동하지 말자 ***** 다항함수와 다르다: y = w1x^2 + w2x + b 에서 x는 몇개인가? 1개 



### Scaling (chapter 3, p.177)

> 통계적인 용어로는, 변수(x), 머신러닝에서는 특성(Feature) 의 범위를 의미
>
> 범위를 축소시키는 것이 일반적
>
> 데이터를 바꾸는 것

standard scalier, minmax scalier 만 보자



#### Normalization(정규화) vs. Standardization(표준화)



연비(mpg): 자동차 무게에 따른 연비 예측



데이터의 특징 확인

통계적으로 보면, 특징은 평균과 분산이다

머신러닝적으로 보면, 모양/분포

자동차의 연비를 예측하기 위해 무게 데이터를 쓰면, 이러한 형태의 모양, 분포를 가지고 있다

x축의 단위를 1/1000로 축소했다고 생각해보자

그래프의 모양이 바뀌는가? 바뀌지 않는다 (단위는 모양에 영향을 끼치지 않는다)

모양은 바뀌지 않지만, 단위를 바꾸면 좋은 점들이 생긴다

	1. 계산량이 줄어든다
	2. 경사하강의 궁극적 목적: MSE를 줄이는 것
	3. x를 여러개 넣어 예측모델을 만들 때, (탑승인원, 자동차 무게 등...), 단위가 여러개 존재하면 단위가 큰 x를 중심으로 학습하게 된다(그래야 MSE가 빠르게 줄어들게 되므로)
	4. 수치가 크다고 꼭 중요한 변수가 아니므로, scaling을 통해 범위를 맞춰준다





#### 데이터 전처리 - Scaling 

- 범위(scale)가 다른 변수들의 범위를 비슷하게 맞추기 위한 목적

- 연속형 변수가 다양한 범위로 존재할 때 제곱 오차 계산 시 왜곡 발생

  - ex) x1은 1에서 10 사이 스케일, x2는 1000에서 100만 사이 스케일

- 스케일이 더 큰 변수에 맞추어서 가중치를 최적화하는 문제 발생

- Scaling in Python

  - from sklearn.preprocessing import MinMaxScaler

  - from sklearn.preprocessing import StandardScaler

    

    

#### Normalization(정규화)

- 변수의 스케일을 0~1 사이 범위로 맞추는 것(Min max scaling)
- 정규화는 변수의 범위가 정해진 값이 필요할 때 유용하게 사용
- X_Normalization = X - min(X) / max(X) - min(X)



#### Standardization(표준화)

- 변수의 평균을 0, 표준편차를 1로 만들어 표준정규분포의 특징을 갖도록 함 (-값이 나올 수 있음)
- 표준화는 가중치(weight) 학습을 더 쉽게 할 수 있도록 함
- X_Standardization = X - mean(X) / std(X)



Normalization, Standardization 해도 데이터의 모양은 변하지 않는다

실제 모델을 사용할 때는 모델을 사용하기 위한 값을 스케일링해서 넣어줘야 한다

머신러닝에서는 크게 영향을 주는 경우가 많지 않으나, 딥러닝에서는 scaling하지 않으면 학습이 잘 안되는 경우가 많다



---

#### 정리

- 스케일링의 목적은 범위를 바꾸는 것. **절대 분포(모양)가 변경되면 안된다**

- 주의) train, test로 나눴을 때, 각각의 최대최소, 평균/표준편차를 사용하면 원래의 모양이 바뀌게 된다

- 따라서 똑같이 train data의 값을 사용해서 스케일링 해야한다 (가장 많이하는 실수)



- 가장 많이 물어보는 질문) 스케일링을 할 때, 왜 train 데이터의 특징으로 test 데이터의 특징까지 처리하는가?
  - 그렇지 않으면 test 데이터의 분포가 바뀌기 때문
  - X_test의 분포가 X_test_scaled의 분포와 바뀌면 안되기 때문

---



### 단일회귀분석과 다중회귀분석



#### 단일회귀분석: Ouput(y)에 영향을 주는 input(x)이 1개인 경우

> 현실적으로 input이 하나인 경우는 많지 않다

- (문제) 습도가 53일 때 불량품 수량 예측
  - formula = 불량품수량 ~ 습도 (y ~ x)



#### 다중회귀분석: Input(x)이 여러 개인 경우(y가 하나, x가 여러개)

> 현실세계에서 일반적으로 보는 문제

- 주로 사용하는 패키지

  - sklearn

    from skelarn.linear_model import LinearRegression

    RA = LinearRegression()

    RA.fit(X_train, y_train)

  - statsmodels

    import statsmodels.formula.api as smf

    Model_1 = smf.ols(formula = 'expenses ~ age + sex' ,

    ​									data = train_set).fit()

  - 어느것을 사용하는지는 상관없으나, 대용량의 데이터를 처리하기에는 sklearn 이 더 효율적이다



- (문제) 제품의 강도는 생산과정의 온도와 시간에 영향을 받음
  - formula = 강도 ~ 온도 + 시간 ( y = w1x1 + w2x2 + b) (y ~ x1 + x2)

- (문제) 의료비 지출 예측 모델링
  - formula - expenses ~ age + bmi + smoker





#### 예측(Regression) 모델 평가지표
  1. MSE: 작을수록 좋음
- 다중회귀에서의 MSE
  - x1, x2 이므로 MSE는 공간상에서 나타나게 된다
  - 공간 상 면과 점들 사이 거리를 최소화시키는 형태로 학습됨
  - 다중회귀에 다항회귀를 섞으면 면이 휘게 된다


2. R square(결정계수): 크면 좋음 (범위: 0~1), 1에 가까울수록 좋음

- y = wx + b // y: mpg(연비), x: weight(자동차의 무게)
- 자동차의 무게와 연비의 관계
  - 자동차의 무게로 연비를 설명할 수 있는가? 
  - 무게로 연비를 어느정도까지 설명할 수 있을까?
  - R-square: 무게로 설명할 수 있는 연비의 "면적"
  - 무게 외에 실린더, 배기량 등을 더 추가하면 설명할 수 있는 면적이 늘어날 수 있다(다항회귀)



- sklearn을 사용하기 위해서 X의 값 전처리가 중요하다
- 전처리의 두가지 방법: Scaling, Encoding
  - Scaling: 적용 전과 후를 비교해보아야 한다





### Encoding

> 숫자를 문자로, 문자를 숫자로
>
> 명목형을 이산형으로 바꿔주는 형태



#### Integer Encoding (p.272)

- from sklearn.preprocessing import LabelEncoder
- 문자형 변수를 숫자형 변수로 변경하여 변수 연산범위를 확대
- 'europe' : 0 , 'japan' : 1 , 'usa' : 2

- 하나의 단어를 숫자로 변환하는 것이 확장되어 자연어처리로 넘어가게 된다



#### One-Hot Encoding

- from sklearn.preprocessing impor OneHotEncoding
- 하나의 값만 True(1)이고 나머지 값은 False(0)인 인코딩

- hot: True를 의미. 하나만 hot하게 한다



일반적으로 분류모델이 예측모델보다 더 어렵고,

실제로 분류 문제는 딥러닝을 사용하는 경우가 더 많다





## 5. Logistic Regression (p.86)

>classification(분류)은 두가지로 나눠진다
>
>- 이진분류(Binary Classification)
>
> - ex) 구매고객이 남자냐, 여자냐 / 정상이냐, 불량이냐/ 음성이냐, 양성이냐
>
>- 다중분류(Categorical Classification)
>
> - 다중분류에 대한 깊은 이야기는 딥러닝에서 다룬다
>
>
>
>분류의 세 파트
>
>1. Sigmoid
>2. Cross Entropy Error
>3. Confusion Matrix



### Logistic Regression



#### Regression: 선형모델을 의미

- Regression의 프로토타입: y = wx + b
- 우리가 하는 분류에 맞지 않음



#### Logistic이란?

- 분류 중에서도 이진분류

- "logit"이라는 확률론의 개념에서 용어가 나와서 Logistic이라고 부름



- 컴퓨터의 입장에서는 남자든 여자든, 정상이든 분량이든 모두 1, 0
- 두개가 서로 다르다는 것. (y가 연속형이 아니라 명목형)



- y_hat = wx + b 에서 y_hat의 범위는?
  - x값의 범위 따라 양의 무한대에서 음의 무한대까지
- 그러나 이진분류 문제에서는  y_hat이 가질 수 있는 값이 0 or 1



- 0~1 사이에 값이 나올 수 있는 것은? 확률(Probability)
- Logistic = Probability(확률)
- y_hat을 확률 범위로 바꿔줄 방법이 필요하다 -  sigmoid
- sigmoid 연산을 취하면, y_hat = sigmoid(wx+b)
  - x값이 아무리 커져도 y는 1보다 클 수 없고, x값이 아무리 작아져도  y는 0보다 작을 수 없게 된다
- Logistic과 sigmoid는 같은 단어라고 생각해도 무방
- Sigmoid(wx + b) => 0~1 (Probability)



### 1) sigmoid()

> 필터 함수, activation(활성화) 함수 등으로 불린다

- sigmoid() 함수를 필터로 사용

- sigmoid(x) = 1 / 1 + e**-x

  - x값의 변화에 따라 y값이 바뀌게 되는 함수
  - e: 오일러상수 (약 2.71)

- sigmoid(0) = 1 / 1 +1 = 1/2 = 0.5

  - x가 0인 지점에 0.5를 지나게 됨

- sigmoid(1000000)

  ​	= 1 / 1 + e**-1000000 (e\*\*-100000은 매우 작은 수, 0에 수렴)

  ​	= 1 / 1 = 1

- sigmoid(-1000000)

  ​	= 1 / 1 + e**1000000 (e\*\*1000000은 매우 큰수, 무한대)

  ​	= 1 / 무한대 = 0에 수렴

- 따라서 sigmoid를 쓰면 y값이 **0과 1 사이**에 나오도록 변형이 된다

​	

---

#### 정리

Logistic Regression이란?

원래 있던 regression에 sigmoid를 씌운 것

Logistic과 Sigmoid는 같은 의미로 쓰인다

---



- sigmoid의 문제점

  - 실제 우리에게 필요한 것은 0 or 1, 그러나 sigmoid는 0~1 사이의 값이 나온다

    - 남자면 남자, 여자면 여자!

    예) Linear Regression: 온도 17도일 때 아아는 몇잔 팔릴까?

    ​	  Logistic Regression: 0 아니면 1



- Classification(범주예측) 모델
  - Output(y)의 수치예측이 아닌 어떤 범주에 속하는지에 대한 예측(확률)을 모델링
- Regression(수치예측) 모델에 Sigmoid(필터) (Activation Function)를 적용하여 구현
  - 일반적으로 분류 기준을 **0.5로 지정**(변경 가능)
  - 0.5 보다 크면 1, 0.5 보다 작으면 0으로 분류
- 분류 결과에 대한 추가적인 신뢰도 검증이 필요(Model Validation)
  - 혼돈 행렬(Confusion Matrix)
  - 정확도(Accuracy), 정밀도(Precision), 재현율(Recall)



- 이 모델 함수를 사용해서 학습을 한다는 것은?

  - 분포된 데이터를 학습하기 위해서 w, b를 바꾸면 모델이 주어진 데이터에 최적화됨
  - sigmoid도 w, b를 조정하면 모델이 데이터에 최적화되어야 할것 (그래프의 모양이 바뀌어야 할 것)
  - **s(wx+b) 함수에 w는 기울기, b는 좌우이동**
    - b는 y축과 만나는 지점. sigmoid 함수의 그래프가 좌우로 움직이면 y축과 만나는 지점이 위아래로 이동함을 볼 수 있음
    - w는 기울기: 최대 모양은 step함수의 모양(S자가 될 수는 없다)
    - sigmoid 함수 안쪽에 있는 w와 b를 학습시켜 모델을 데이터에 튜닝
- 오차가 줄어드는 방향으로 update 하는 방식은 똑같이 "Gradient Descent", 다른 식의 오차함수(Cross Entropy Error)를 사용하게 됨
  

  



### 2) Cross Entropy Error(CEE)

> 딥러닝에서는 중요한 개념
>
> 
>
> **Machine Learning Modeling**
>
> 1. Regression
>
>    - Learning(Training) -> (Train Data의) MSE 
>
>      - mean(y-y_hat)**2: 실제값과 예측값의 차이
>
>    - Validation -> (Test Data의) MSE
>
>      (학습과 검증에 모두  MSE를 씀)
>
> 2. Classification
>
>    1) 이진분류(Binary Classification)
>
>    - Learning -> 과연 무엇을 보며 에러를 낮출 것인가? 
>
>      - MSE가 가능은 하지만 효율적이지 않다. 
>
>      - **Cross Entropy Error(CEE)**를 사용
>
>    - Validation -> Accuracy, Precision, Recall, F1 Score
>
>    2) 다중분류(Categorical Classification)



#### Entropy

- Entropy란? "불순도" 라고 생각하자

- "불순도"가 작으면 좋은 것

  예) 파란색 점 무리, 빨간색 점 무리가 있음

  ​	  파란색은 파란색끼리만, 빨간색은 빨간색끼리만 있다면? 불순도가 없다

  ​	  파란색과 빨간색이 섞여있다면? 불순도가 생겼다

- 불순도가 낮아지는 쪽으로 모델을 학습시키겠다(w,b값을 바꾸겠다)

- 어떻게 측정할 것이냐의 문제



- Entropy 를 이해하기 위해서는 정보 이론을 알야아 한다



#### Information Theory(정보이론)

- **Information Gain(정보이득량)**

  - 1) 자주 발생하지 않는 사건은 2) 자주 발생하는 사건보다 전달하는 **정보량**이 많음

    - 1) 발생확률이 낮은 사건, 2) 발생확률이 높은 사건

    - Information Gain은 정보의 희귀성(발생가능성)에 반비례

    - 언제 발생할지 모르므로, 자주 발생하지 않는 사건이 더 많은 정보를 준다

      예) 신용카드의 상환과 연체

    - I(x) = -log(P(x))

      - I: Information
      - P(x): probability x (사건의 발생 확률)
      - 1/P(x) 에 log를 씌워 -log(P(x))가 됨

- Degree of Surprise(놀람의 정도)

  - 예상하기 어려운 정보에 더 높은 가치를 매기는 것



- 이것을 entropy로 변환시키면,



#### Entropy(불순도 Metric척도)

- 정의: 불순도의 정도
- Entropy = E(-log(P(x)))
  - E: Expected Value(기댓값), 기댓값이란? 확률의 평균
  - -log(P(x)): 정보량
- 확률변수의 평균 정보량(기댓값)
  - -sum(P(x) * log(P(x)))
    - P(x)와 log(P(x))는 같은 P(x)로 계산
  - 놀람의 평균 정도

- 불순도가 낮으면 분류정확도가 높아짐



#### Cross Entropy Error

- 왜 cross entropy라고 하는가?

  - entropy를 계산식을 보면 **-sum(P(x) * log(P(x)))**: P(x)가 같다

  - cross entropy error의 식을 보면, **-y * log(y_hat) - (1-y)*log(1-y_hat)**

  - P(x)를 같은 값을 쓰지 않고, 앞에 있는 확률값과 뒤에 있는 확률값을 교차해서 계산하므로 "cross" entropy

    (y와 y_hat의 차이점을 보려고 하는 것이라서)

- 서로 다른 사건의 확률을 곱하여 Entropy를 계산
- y를 Cross-Entropy의 가중치로 적용



#### Binary Cross Entropy Error

- -y * log(y_hat) - (1-y)*log(1-y_hat) : 실제로는 y=0, y=1일 때를 더해놓은 것

- 왜 이렇게 생겼는가?

  - 이진분류에서  y가 가질 수 있는 값은 0, 1밖에 없다

  - y = 0 일 때

    -(1-y)*log(1-y_hat)

    = **-log(1-y_hat)**

  - y = 1일 때

    -y*log(y_hat)

    = **-log(y_hat)**

  

- y = 1일 때: CEE = -log(y_hat)

  - y_hat은 0~1사이 값밖에 나오지 않음
  - y가 1일 때 y_hat을 1이라고 했다면 CEE(오차)는 0
  - y가 1일 때 y_hat을 0이라고 했다면 CEE(오차)는 무한대로 커진다


- y가 0일 때: CEE = -log(1 - y_hat)

  - y_hat에도 -가 붙어있으므로 -log 그래프가 좌우대칭된다
  - y가 0일 때 y_hat을 0이라고 하면 CEE는 0
  - y가 0일 때 y_hat을 1이라고 하면 CEE는 무한대로 커진다

  

- y값에 가까워질수록 값이 줄어들도록 만든 것이 CEE



- CEE를 쓰면 y=0, y=1 를 구분해서 오차가 얼마나 발생했는지를 정량적으로 측정할 수 있다
- 분류 문제에서는 MSE 보다  CEE를 쓰는 것이 학습효과가 좋고 학습효과도 빨라진다
- 경사하강법을 쓸 때 log함수를 미분해야 한다
- CEE = -log(1 - y_hat) = -log(1 - sigmoid(wx+b))





- 오차(CEE)를 가지고 성능을 평가할 수 없다





### 3) Confusion Matrix (p.356)

> - 분류모델의 Validation
> - 이진 분류 문제에서 y는 0 또는 1
> - y_hat 값은 0 ~ 1 사이 값. 따라서 0.5를 기준으로 0 또는 1로 바꾸어줌
> - y와 y_hat이 잘 맞는지 교차해서 비교한 것이 Confusion Matrix



#### 이진 혼돈 행렬(Binary Confusion Matrix)

- 이진분류는 알고싶은 것을 0(Positive)로, 상관없는 것을 1(Negative)로 둔다

|               |             | y_hat<br />(예측)     |                       |
| ------------- | ----------- | --------------------- | --------------------- |
|               | 비교        | 0(Positive)           | 1(Negative)           |
| y<br />(실제) | 0(Positive) | O(True Positive, TP)  | X(False Negative, FN) |
|               | 1(Negative) | X(False Positive, FP) | O(True Negative, TN)  |

- 제대로 예측했는지 교차행렬표로 표현
- False Negative: 잘못된 Negative (True인데 Negative로 표시함)
- False Positive: 잘못된 Positive (False인데 Positive로 표시함) 



#### Accuracy(정확도)

- (문제) 신용카드 데이터

- Positive(상환)와 Negative(연체)로 맞게 분류된 데이터의 비율
- (TP + TN) / (TP + TN + FP + FN) = 2921명 / 3000명 = 97%
- 정확도가 높은가?  YES
- 좋은 모델인가? 모른
- 정확도를 가지고 좋은 모델인지 아닌지를 알 수는 없다



#### Precision(정밀도)

- Positive(상환)로 분류된 결과 중 실제 Positive(상환)의 비율
- Spam메일이 알고싶은 것이므로 Positive(0), Ham메일이  Negative(1)
- Negative(=Ham mail)를 Positive(=Spam mail)로 틀리게 분류 시 문제 발생: 스팸메일 필터링
  - Spam -> Spam(o)
  - Ham -> Ham(o)
  - Spam -> Ham(x): 그냥 불편한 것
  - Ham -> Spam(x): 비즈니스 임팩트가 커짐
- 이 경우 Precision가 높은 모델을 만들어야 한다



#### Recall(재현율)

- 실제 Positive(상환) 중에 Positive(상환)로 분류된 비율
- Positive를 Negative로 틀리게 분류 시 문제 발생: 코로나 진단
- 양성환자를 알고 싶은 것이므로 양성(Positive), 음성(Negative)
  - 코로나 걸렸는데 -> 코로나에 걸리지 않았다고 함: 비즈니스 임팩트가 더 큼
  - 코로나에 안걸렸는데 -> 걸렸다고 함
- 이 경우 Recall이 높은 모델을 만들어야 한다
- = 민감도(Sensitivity), 적중률, 진짜양성비율



- Business Impact
  - Negative impact를 의미
  - 어떤 일에 부정적 영향





(문제) 신용카드 회사에서는 상환보다 연체하는 사람이 더 알고싶음

- 연체가 Positive, 상환이 Negative
- Precision, Recall 중 중요한 것? (이진분류에서는 정확도보다 둘이 더 중요한 경우가 많다)
- 문제점: 상환자가 연체자보다 압도적으로 많다
- 모델의 성능에 상관없이, 비즈니스 특성 상 "전부 다 상환합니다" 라고 해도 라고 해도 정확도 97%
- 비즈니스적인 관점에서 정확도가 별 의미가 없게 된다
- 데이터가 한쪽으로 쏠려있는  경우는 좋지 않다



#### Positive(연체) 기준

- 정확도: 2921/3000 = 97% 바뀌지 않는다
- 정밀도: 32 / 39 = 82%
- 재현율: 32 / 104 = 31%



- 연체(Positive)인데 -> 상환(Negative)할 거라고 예측: 비즈니스 임팩트가 더 큼
- 상환(Negative)인데 -> 연체(Positive)할 거라고 예측

- 재현율이 더 중요



- 정확도, 정밀도, 재현율 자체가 중요한 게 아니라, 좋은 모델을 만들기 위해서는 비즈니스에 대한 이해가 있어야 한다
- 분류모델 측정에는 많은 고민이 필요하다





---

#### 정리

- y와 y_hat의 맞음과 틀림 정도를 계산하는 것

- 정확도 외 precision, recall 등으로도 계산할 수 있다

- Binary Confusion Matrix를 만들고, 내가 더 알고 싶은 것을 Positive로 두어야 한다

  (이진분류의 경우 내가 알고싶은 것이 존재하는 경우가 많다)

- 정확도: Positive를 Positive로, Negative를 Negative로

- 정밀도: Positive 분류된 것 중 진짜 Positive 비율

- 재현율: 진짜 Positive중 Positive로 분류된 비율

---



- 어떤 경우는 positive, negative 상관없이 모두 잘 맞추는 게 중요할 수도 있다
- 데이터의 비율이 비슷할 경우 정확도, 아니라면 F1-Score 사용 가능 



#### Evaluation Method: F1-Score(0에서 1 사이 값)

- 정밀도와 재현율은 **Trade-off관계** (둘다 높일 수는 없다): 통계학의 1종 오류와 2종 오류의 관계
- F1-Score: 정밀도와 재현율의 조화평균
  - 조화평균: 정밀도와 재현율 같은 비율에 대한 평균을 구하기 위해서는 조화평균 사용
  - 일반적으로 사용하는 평균은 산술평균

- F1-Score = 2 / 1/Precision + 1/Recall = 2 * ((Precision * recall) / (Precision + Recall))



p.357

에러의 종류

잘못 분류한 샘플의 수가 원하는 정보의 전부느 아니므로, 정확도만으로 예측 성능을 측정하기에는 부족할 때가 종종 있음.





#### (참고) ROC, AUC: 이진분류에만 사용되는 비율계산 (p.376)







---

#### Logistic Regression 정리

- 데이터를 숫자로(0 or 1) 인코딩하여 처리
- Regression Analysis는 y_hat의 범위가 -무한대~무한대
- 분류 문제에서는 y는 sigmoid 활성화함수를 쓰므로, y_hat = sigmoid(xw+b) 으로 0~1 사이 값만 나오게 된다
- sigmoid는 확률의 "logit"이라는 개념에서 온 것으로, logistic = sigmoid 라고 생각해도 무방하다
- 우리가 필요한 것은 0 or 1 이므로, 분류기준을 0.5로 하여 1, 0으로 처리하게 된다 (sklearn predict에 자동으로 처리)
- sigmoid 안의 b, w 값을 튜닝하여 위치와 기울기를 조정하여 학습
- 경사하강할 때 쓰는 오차함수가 CEE
  - CEE = -y*log(y_hat)  - (1-y)\*log(1-y_hat)
- 두가지 다른 확률을 사용하므로 cross entropy
- 학습을 평가할 때는 accuracy, precision, recall
- 비즈니스에 어떤 영향을 주는게 더 심각한 것인지(비즈니스 임팩트)를 파악하는 것이 중요
- 그에 따라 precision, recall을 선택해야 함 (이진분류에서 정확도를 쓰는 경우는 많지 않다)
- recall에 대한 문제를 푸는 경우가 더 많긴 하나, 비즈니스에 따라 다르다
- Confusion Matrix를 제대로 그리는 것이 중요하다
- Precision과 Recall을 함께 측정하고 싶다면 F1 Score(조화평균)을 사용한다
- 모델을 생성할 때는 sklearn의 LogisticRegression을 사용한다
  - 여러 파라미터가 있으나, 전체적인 모델을 본 이후 나중에 한번에 정리할 예정

---





## 6. Decision Tree(의사결정 나무)

> 지도학습 알고리즘의 하나 (x, y 필요)
>
> Machine Learning에서 Machine이란?
>
> - 함수형 모델(y = wx + b): 학습의 대상이 파라미터(w, b)
> - 나무형 모델: 학습의 대상이 Ruleset(분류기준)
>   - 분류기준(Rule1, Rule2, Rule3, Rule4, ...)을 찾아내야 한다
>
> - 머신러닝에서는 나무계열의 모델로 거의 수렴하고 있다
> - 딥러닝은 함수형 모델이다



### 1) Decision Tree 개요

(p.101 ~)



예) y = 녹색 or 회색, 이진분류

- 이진분류를 여러번
- 첫번째 기준: X1 > 0.5 이면 True, 아니면 False
- 두번째 기준: X2 > 0.5 이면 True, 아니면  False
- 세번째 기준: X1 > 0.25 이면 True, 아니면 False



- 이 나무를 찾아내야 함 (스무고개와 같다)



- 어떤 기준으로 0.5에서 자르게 되었을까?
- 함수형 모델의 학습원리: w, b를 찾는 학습의 원리는 경사하강법
- 나무형 모델의 학습원리: 분류규칙은 알 수 없으나, 새로운 나뭇잎이 들어왔을 때 이 나뭇잎이 회색인지, 녹색인지 분류하기 위해 모델에 적용해봄

---

- 나무를 만들어, 나무로 예측과 분류를 하는 것을 "Decision Tree", 나무형 모델이라고 한다
- 나무의 끝에 어떤 레이블을 갖는지 조건(Rule1, Rule2, Rule3, Rule4, ...)을 찾아내야 하는 것
- 왜 나무라고 하는가?
  - 제일 위를 root, 가운데를 branch, 제일 끝을 leaf 라고 한다
  - 거꾸로 자라는 나무모양. "나무가 자란다"고 표현한다(depth)

---



- 가능한 대답이 두가지인 **이진 질의(Binary Question)**의 **분류 규칙**을 바탕으로 최상위 루트 노드(Root Node)의 **질의 결과**(Rule)에 따라 가지(Branch)를 타고 이동하며 최종적으로 분류 또는 예측값을 나타내는 리프(Leaf)까지 도달
  - 범주형 자료: Classification Tree(분류 나무)
    - 분류일 때는 entropy를 기준으로 Rule을 찾아낸다
    - sklearn DecisionTreeClassifier
  - 수치형 자료 : Regression Tree(예측 나무)
    - 비슷한 숫자들끼리 모이게 하고싶으므로, 분산이 작아지는 형태로 split한다
    - sklearn DecisionTreeRegressor
- Rule 기반 의사결정 모델 ==> 나무
  - ex) Rule: IF 흡연 > 5년 AND 운동량 <3회 THEN "위험"
    - Rule1: 흡연 > 5년 - True
    - Rule2: 운동량 < 3년 - True
    - Leaf: "위험"
    - 기준을 데이터로부터 스스로 학습해야 한다



- 왜 저렇게 잘랐을까?

  - 기준: **불순도**를 최소로 하는 지점을 찾아냄 (데이터에 의해 스스로 학습함)
    - 불순도(에러)는 낮아지고, 순도(accuracy)는 높아지는 방향
    - 쪼개본 것 중에 불순도가 제일 낮은 곳을 결정함(학습함)
  - 데이터가 주어지면, 불순도가 최소인 지점을 찾아낸다
  - train, test로 갈라 train으로 기준을 잡고 test해봄

  


#### Root Node: 최상위 노드

- Splitting:하위 노드로 분리되는 것
- Branch: 노드들의 연결(의사결정나무의 일부분/Sub-Tree)



#### Decision Node: 2개의 하위 노드로 분리되는 노드

- Parent Node: 분리가 발생하는 노드



#### Leaf(Terminal Node): 더이상 분리되지 않는 최하위 노드

- Child Node: 분리가 발생한 후 하위 노드





- 규칙 기반으로 직관적으로 이해하기 쉽고 설명력이 좋은 알고리즘

  - 각 노드 별로 **불순도(Impurity)**에 기반한 최적의 분류 규칙(Rule)을 적용
    - 엔트로피를 낮추는 방향으로
    - 불순도를 엔트로피를 이용해 계산함
  - 분리(Splitting) 과정을 반복하면서 **1) 의사결정나무가 성장**
    - 1)의 의미: (1) Model Capacity가 좋아지면서 (2) Overfitting의 위험성이 올라감
  - 각 리프(Leaf)는 **동질성**이 높은(=순도가 높은) 적은 수의 데이터포인트를 포함

  

- 동질성이 높은 그룹 구성을 위해 **재귀적 파티셔닝(Recursive Partitioning)** 수행

  - 1단계: 동질성이 높은 두 그룹으로 나눌 수 있는 **이진 질의** 적용
  - 2단계: **종료조건**을 만족할 때까지 1단계를 반복
    - 기본 종료조건: 불순도가 없어질 때까지
    - criterion = 'gini'를 'entropy'로 바꿀 수도 있다 (sklearn은 기본으로 gini 사용)





- 설명력이 가장 좋은 모델

- 함수형보다 성능이 더 좋은 경우도 많다

  

- 기존에 있던 개념으로 학습의 원리를 적용하는 것이 머신러닝, 딥러닝이다

- 모든 인공지능은 결국 오차를 최소화하는 것이다

- 우리가 하는 학습도 결국... 데이터와 우리의 오차를 줄여나가는 게 아닐까?





- Model Capacity: 모델의 데이터에 대한 설명력

  ex) 함수형에서, 

  ​	y = wx + b

  ​	y = w1x^2 + w2x + b

  ​	y = w1x1 + w2x2 + b

  - 함수의 학습대상인 파라미터의 개수가 많으므로 설명력이 높다

  ex) 나무형에서,

  - Rule의 개수가 Model Capacity에 직접적인 영향을 준다
  - 나무가 깊게 자라면 자랄수록 Capacity가 좋아지고, Data에 대한 설명력/표현력이 좋아진다

  

  - 함수형에서도 파라미터의 개수가 지나치게 많아지면 overfitting 발생 가능성 높아지는 것처럼

  - 나무형에서도 Rule이 지나치게 많아지면 overfitting이 발생할 수 있다

    (이론상으로는 entropy = 0 될 때까지 자라날 수 있다)

    - 좋은 모델이지만 overfitting이 발생할 수 있는 문제점이 큰 모델이기도 하다
    - max_depth를 통해 제약을 걸어줄 수 있다

  



#### 과적합(Overfitting) 문제

- 각 노드는 동질성이 높고 불순도(Impurity)가 낮은 방향으로 분리
- 너무 **복잡하고 큰 의사결정나무 모델**을 생성하여 과적합 문제 발생



#### 대응방법 - 가지치기(Pruning)

- 모델 성능 향상 및 과적합 예방 목적
- max_depth: 의사결정나무의 **성장 깊이**를 지정
  - 사용자가 지정해줄 수 있음(하이퍼 파라미터)
- min_samples_leaf: 리프에 들어가는 **최소 샘플의 개수** 지정





### 2) Entropy vs. Gini Impurity Index

> 의사결정나무의 학습의 원리: 불순도가 낮아지는 방향으로
>
> 불확실성, 불순도를 측정하는 방법도 하나가 아니다
>
> - sklearn의 default 불순도 체크 방식은 Gini Impurity Index(지니 계수)
>
> - R의 default는 Entropy
>
> 어느 것을 쓰는 게 더 좋을까? 해봐야 안다^^
>
> 거의 차이는 없지만, 성능상 미묘한 영향을 줄 수 있다



"계산하는 방식은 Entropy, Gini Impurity Index가 있다"는 것만 기억하자!

계산하는 방식 두가지(참고만)



#### 학생 vs. 소득

|      | 학생(x1) | 소득(x2) | 연체(y) |
| ---- | -------- | -------- | ------- |
| 1    | 네       | 없음     | Yes     |
| 2    | 아니오   | 없음     | No      |
| 3    | 네       | 없음     | Yes     |
| 4    | 아니오   | **있음** | No      |
| 5    | 네       | 없음     | Yes     |
| 6    | 아니오   | 없음     | No      |



- 학생은 무조건 연체: 불순도 없음
- 소득이 없는 경우는 y_hat의 경우에 불순도가 발생(Yes, No가 섞이게 됨)



#### Entropy

- -sum(p(x)*log(p(x))): 정보량과 확률을 곱한 것의 합

- log의 밑이 e이든 2이든 크게 상관없다(크기를 비교하는 데에는 전혀 상관 없음)
- 분리 정보 이득이 **큰 특징**으로 분리 발생
- 분리 정보 이득 = 질문 전 Entropy - 질문 후 Entropy(일반적으로 말하는 불순도)



- 질문 전 Entropy

  - -p(Yes) * log(p(Yes)) - p(No) * log(p(No))

    = -p(3/6) * log(p(3/6)) - p(3/6) * log(p(3/6))

    = -0.5 * log(0.5) - p(0.5) * log(0.5) = 1

  - 아무것도 질문하기 전의 Entropy 는 1

- (소득) 질문 후 Entropy

  - p(있음) * E[Yes, No] + p(없음) * E[Yes, No] = 0.996

    = (1/6) * E[0, 1] + (5/6) * E[3, 2]

    = (1/6) * (-(0/1) * log(0/1) - (1/1) * log(1/1)) + (5/6) * (-(3/5) * log(3/5) - (2/5) * log(2/5))

    = 0.996

- (소득) 분리 정보 이득: 1 - 0.966 = 0.034

  - 낮은값... 별로 안좋음

  

- (학생) 질문 후 Entropy

  - p(네) * E[Yes, No] + p(아니오) * E[Yes, No]

    = (3/6) * E[3, 0] + (3/6) * E[0, 3]

    = (3/6) * (-(3/3) * log(3/3) - (0/3) * log(0/3)) + (3/6) * (-(0/3) * log(0/3) - (3/3) * log(3/3))

    = 0

- (소득) 분리 정보 이득: 1- 0 = 1

  - 아주 높음!



#### Gini Impurity Index

- 분리정보이득과 반대 (지니 불순도 계수는 낮은 게 좋다)

- 1 - sum(p(x)^2) (1-각각 사건의 확률값의 제곱의 합)
- 특징 지니 지수가 **클수록** 불순도(Impurity)가 낮음
  - 특징 = 변수, feature

- 지니 불순도 지수가 **작은 특징**으로 의사결정 나무의 노드를 분리



#### 지니 지수 적용

- 두 개의 특징으로 분리된 각 노드의 지니 지수를 계산
- 두 개의 결과를 사용하여 **특징 지니 지수 계산**
- 지니 불순도 계수(0~0.5 사이의 값) = 1-특징 지니 지수





- 모델을 작게 만들었을 때 동일한 성능을 낸다면, 굳이 복잡한 모델을 쓸 필요가 없다
- 최근 트렌드: 비슷한 성능을 내면서 가벼운 모델을 만드는 것
  - 함수형: 파라미터 개수를 줄이고, 나무형: 깊이를 줄이는 것









## 7. Random Forest

결정트리의 앙상블

