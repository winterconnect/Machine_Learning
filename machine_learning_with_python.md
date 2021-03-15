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



#### Recall(재현율) = Sensitivity

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
  - X가 0.5보다 크니? Yes, No => Y가 0.5보다 크니? ==> ...

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
    - 불순도 계산법: Entropy, Gini
    - Terminal node에는 동질성이 높은(똑같은 값만) 값만 들어가도록

    

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

    - 좋은 모델이지만 overfitting이 발생할 수 있는 문제점이 큰 모델이기도 하다 (가장 큰 문제)
    - max_depth를 통해 제약을 걸어줄 수 있다

  



#### 과적합(Overfitting) 문제

- 각 노드는 동질성이 높고 불순도(Impurity)가 낮은 방향으로 분리
- 너무 **복잡하고 큰 의사결정나무 모델**을 생성하여 과적합 문제 발생
  - Train Error, Test Error의 차이가 크게 나면 과적합을 의심해야 함



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

계산하는 방식 두가지(참고만): 한번씩은 손으로 계산해볼 것을 권장



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





#### Feature Importance

- 우리가 집어넣은 X중 어떤 것이 기여도가 큰지(분류하는 데 더 많이 사용되었는지) 특징 정보를 볼 수 있음
- 트리 계열의 모델은 보통 feature importance를 제공해준다





Data Scientist

- 컴퓨터 사이언스
- 수학&통계학 지식
- 비즈니스 분석능력: 인간 지적노동하던 부분을 AI를 통해 해결해내겠다 는 것이 목표
- 비즈니스 분석이 없는 AI는 없다





## 7. Random Forest

> 나무모델의 확장



결정트리의 앙상블 (p.115)



#### Ensemble

- "조화를 이루다"는 프랑스어

- (음악용어) 연주할 때 하나의 악기를 사용하는 것보다 여러개의 악기를 사용하면 더 아름다운, 좋은 소리를 만들어낼 수 있다

- 머신러닝에도 앙상블이라는 기법이 존재하고, 하나의 데이터를 처리할 때 하나의 알고리즘이 아니라 여러 개의 알고리즘을 함께 사용하여 결과를 도출하는 것
- 결과를 조합해서 새로운 형태의 모델을 만들어내는 것을 "앙상블"이라고 한다

- 서로 다른 타입의 모델을 섞어야 함(여러 악기를 섞는 것이지, 바이올린 100개, 비올라 100개를 섞는것을 앙상블이라고 하지는 않는다)



#### Regression을 풀기 위해

​	1) 선형모델(Linear Regression) 2) 나무모델(Regressor) 3) 신경망모델 등 다양한 방법을 쓸 수 있다



#### Classification을 풀기 위해

- 데이터에 적용했을 때, Logistic Regression vs. Decision Tree 중 DT가 더 성능이 좋았다

- Random Forest란? 결정트리의 앙상블

  - forest: 나무가 많은 것... 나무형 모델만 많이 쓰는 것

  - classification을 풀기 위해 tree 1, tree 2, tree 3... 나무만을 조합해 새로운 결과를 만들겠다는 것

    (sklearn을 쓰면 나무형 뿐만 아니라 여러 기법을 섞는 것이 어렵지는 않다)



#### Forest의 의미

1. 나무가 여러개: 동일한 데이터로 나무를 만들면, 나무가 다 똑같이 생기지 않았을까?
   - 똑같은 나무를 만들어봤자, 의미가 있을까? 없다
   - 여러 나무가 다양한 관점에서 데이터의 특징을 설명해주길 바라는 것
2. 다양한 모양으로 생성
   - 앞에서 배운 방식으로는, 많이 만들어봤자 똑같이 생긴 모델이 만들어짐(entropy를 계산하므로)
   - 어떻게 다르게 만들 수 있을 것인가?



- 결국 데이터에 의해 나무의 모양이 만들어짐

- 나무를 만드는 방법을 바꿀 수가 없으므로, 데이터가 같다면 나무는 같은 형태로 계속 만들어진다

- 다양한 형태의 나무를 만드는 방법은, 데이터를 다르게 만들면 된다. 어떻게?



- 데이터가 다양하면서도, 원본 데이터의 특징을 담고 있어야 한다

- 데이터를 변형시킬 것(재구성할 것)

- 재구성하는 과정에서 사람이 개입하면 안된다

- 원본 데이터의 특징을 포함하는 형태로 재구성하는데, 분석자의 의도가 반영되지 않는 형태로 "랜덤"하게 재구성함



예) iris

| sepal_length | sepal_width | petal_length | petal_width | y    |
| ------------ | ----------- | ------------ | ----------- | ---- |
|              |             |              |             |      |

- Train(학습용데이터): 105개 * 4 (y 제외)

  - 데이터를 바꾸는 방법?

    1) 행의 개수를 바꾼다: 105개보다 작게 쓴다

    2) 열의 개수를 바꾼다: 4개 중 일부를 사용한다





- **덜 정확한 분류모델** 여러 개를 모아서 더 정확한 분류모델을 만들 수 있을까?

  - 앙상블(Ensemble): 여러가지 모델을 사용하여 **정확도를 개선**하는 방법

  

- 모델의 성능을 좋게하는 가장 좋은 방법은 데이터를 많이 넣는 것이다.

  - 데이터의 특징을 확인하는 것이므로 데이터가 많을수록 모델의 성능은 높아진다

  

- 105개보다 많이 쓸 수는 없으므로 105개보다 적게 쓴다

  - 일반적으로 105개로 만든 모델보다는 일반적으로 성능이 떨어진다

- 그렇게 만든 모델을 취합하면, 전체 데이터를 쓴 것보다 성능이 좋아진다는 것이 랜덤 포레스트의 이론



- 학문적이 아니라 실험적으로 입증되며 만들어진 모델
- 미국에서 범죄발생 예측하는 모델을 만들어 테스트 할 때, 의사결정나무를 만들어보고, 랜덤포레스트를 만들어보니 성능이 더 좋아짐
  - 1000개 정도의 의사결정나무를 만들어 테스트
  - 1개의 랜덤포레스트를 만들어 비교하니, 성능이 더 좋은(비슷한) 의사결정나무는 1000개 중 3개밖에 없었다



- 로지스틱 < 의사결정나무 < 랜덤포레스트

- 의사결정나무: 하나의 좋은 나무를 만들려는 것



- 랜덤포레스트: **의사결정나무의 앙상블**(Ensemble)

  - 다수의 의사결정나무들의 결과로부터 모델을 생성

  - 모델 생성에 **다양성**(Diversity)과 **임의성**(Random)을 부여

    - 다양성/임의성: 서로 다른 나무를 만들려고 하는 것

  - 모델 정확도를 높이고 **과적합 발생 가능성을 낮춤**

    - 의사결정나무 모델의 최대 단점을 보완할 수 있음

  - 올바른 예측은 **강화**하고, 잘못된 예측은 **상쇄**하는 경향 존재

    - 부스팅 계열의 모델이 이렇게 동작

    

- 최근에는 랜덤포레스트보다 부스팅계열의 모델을 쓰는 것이 추세
  
  - 특히 XGBoost (반드시 좋은 것은 아니다)



- 데이터의 다양한 부분을 다양한 모델로 학습해서 정확도를 높이는 것



- 다양성과 임의성을 높이는 방법 중, 다양성을 높이는 방법 부터 살펴보자





### 1) 다양성(Diversity)

> 행을 건드려서 서로 다른 나무를 만드는 방법



#### 다양성 - 배깅(Bagging)

- 통계학에 원래 있는 개념: 분석 데이터를 재구성하는 방식

- Bagging = Bootstrap + Aggregating

  - Bootstrap: 새로운 데이터셋을 만들어냄

- 주어진 데이터를 사용하여 **여러 개의 서로 다른 Train Data**를 생성

  - 생성된 Train Data(=Bootstrap Data)마다 별도의 의사결정 나무 모델 생성

  - Hyperparameter(n_estimators)로 의사결정 나무 개수 지정

    (몇개를 만들면 좋은지는 해봐야 안다)

    - sklearn n_estimators default: 100개 
    - 현업에서는 최소 5000개, 최대 25000개까지... (시간이 오래 걸림)

- 개별 Train Data는 Bootstrap 방식으로 생성

  - Bootstrap Data는 Original Data에서 **단순복원 임의추출법**으로 생성

    

- 단순 복원 임의추출법: 105개 중 일부 데이터를 샘플링해서(통계용어: 표본추출) 새로운 데이터셋을 만듦
  - 뽑았던 데이터를 다시 넣어 뽑음(다시 뽑힐 수도 있고, 아닐 수도 있음)
  - 원본데이터의 특징을 가지고 있다고 봄
  - 랜덤샘플링 하면 분포가 비슷하게 추출됨
  - 원본 나무의 모양과, 만들어진 나무의 모양은 다를 수 있음
  - 서로 다른 형태의 나무가 만들어지고, 나무의 결과를 취합하는 과정(Aggregating)이 필요해짐



인공지능은 장비가 좋아야 한다...

남들이 10번 돌릴때 나는 1번 돌리면 학습을 그만큼 못하는 것

대부분의 모델링은 해봐야 아는 것이 많기 때문에 학습을 많이 할수록 좋은 성능을 낸다





#### 다양성 - Bootstrap

- 다양한 데이터를 만들어냄

- Original data vs. Bootstrap data

- n_estimators = 3
- y값을 예측하는 것
- 나무가 3개면 예측값이 다르게 나올 수 있을 것
- 서로다른 예측값을 통합해서 하나의 의견으로 만드는 과정이 필요



#### Aggregating

- Bootstrap으로 만들어진 결과를 취합

- 여러개의 Bootstrap모델의 결과를 통합

- 분류모델: **다수결** 또는 가중치를 적용하여 통합

  ex) y_hat = {1, 0, 0, 0, 1, 1, 1, 1, 1} = 1 (다수결)

  - 각 행마다 y값은 bootstrap 모델만큼 나올 것 - 다수결

  - 100개중 50:50이 나오면 어떡하나? 확률적으로 그러한 가능성은 거의 없으므로 홀수개로 나무를 만들지 않아도 무방하다

    

- 예측모델: **평균값** 또는 가중평균값으로 통합

  ex) y_hat = {77, 75, 76, 77, 76} = 76.2





### 2) 임의성(Random)

> 열을 건드려서 다양한 데이터를 만드는 방법



- feature importance 를 통해 분석가가 선택할 수 있는 영역
- feature를 임의로 선택되게 하는 것이 "임의성"



- split이 발생할 때, 어느 feature가 entropy를 제일 낮추느냐가 중요함

- 그런데 만약, 한 feature가 없다면?

- 나머지 3개 중 수학적으로 entropy가 가장 낮은 feature를 고를 수 있음

- 임의성은 이 3개를 고정시키지 않겠다는 개념



- 4개를 다 사용하는 것이 아니라, feature들을 부분적으로 사용하며 사용할 feature를 랜덤으로 선택

- 원래 있던 x값에서 subset를 뽑아내고, random subspace를 생성



#### 임의성(Random) - Random Subspace

- 나무가 만들어지는 과정에 random subspace가 발생함

- 의사결정나무 생성 시 변수(x) 무작위 선택

- 원래 변수에서 무작위로 입력 변수를 추출하여 적용

  - 변수: 통계에서 variable, 머신러닝에서 feature

- 무작위 입력변수의 개수를 **1~전체 변수의 개수** 사이에서 지정

- Hyperparameter: max_features

  - 기본값: sqrt(변수의 개수) ex) 변수가 4개면 2개

- max_features = 3

  ex) 9개 중 3개 지정 (기본으로 3이 지정됨)

  - split이 발생하는 시점에 9개 중 3개가 랜덤하게 뽑힘(3개는 비복원추출)
  - 3개 중 entropy를 제일 낮추는 변수로 split 발생
  - 다음 split 때 다시 9개 중 3개를 뽑음(복원추출)



(통계학에서는 hyperparameter를 "모수", "초모수"로 번역하는 경우도 있으나 그냥 hyperparameter로 익히자)



- p.120, 110 기여도 비교
  - 임의성을 쓰면 x가 좀 더 다양하게 활용됨을 확인할 수 있다

  - Random Forest를 보면 feature importance가 좀 더 다양하게 참조된다

  - 못봤던 특징을 다양한 관점에서 볼 수 있게 된다

    (머신러닝 시장의 주요 흐름은 랜덤 포레스트인 것이 바로 이러한 이유)

 



#### Hyperparameter Tuning

- n_estimators: 모델에 사용되는 의사결정나무의 개수
- max_features: 분할에 사용되는 Feature의 개수
- max_depth: 트리모델의 최대 깊이를 지정



- max_leaf_nodes: 말단 노드의 최대 개수
- min_samples_split: 분할을 위한 최소한의 샘플데이터 개수
- min_samples_leaf: 말단 노드가 되기 위한 최소한의 샘플데이터 개수



보통 위의 3개 하이퍼파라미터를 튜닝한다

그렇지 않으면 너무 많은 경우의 수가 발생한다

이것을 처리하기 위해 sklearn은 GridSearchCV() 를 제공한다

(다양한 경우의 수를 계산)



- GridSearchCV() in sklearn (p.337)



### 3) Cross Validation (교차검증)

(p.324)

>model validation 으로 다시 돌아가보자
>
>- 모델을 만들 때,
>- capacity가 다른 여러 모델 중 어느 것이 가장 잘 설명하느냐를 보았음
>- 정량적인 수치로 측정하기 위해 MSE를 사용했음
>- training error: 생성과 평가에 같은 데이터를 쓰면 overfitting의 문제 발생 가능
>- 일반적인 성능을 내는 모델을 만들기 위해 testing error를 사용(training, testing data로 쪼개서 사용)
>
>
>
>- training data로 fitting
>- 여러 개의 모델이 만들어지면, testing data로 모델을 평가
>  - m.predict(X_test)로 y_hat 생성, mse나 accuracy로 validation
>- model selection 하는 것도 학습의 일부로 보기 때문에 validation data를 만들었음
>  - m.fit(X_train, y_train)
>  - m.predict(X_validation) 으로 모델 평가
>  - 성능이 가장 좋았던 모델 하나로 generalization error를 추정
>- training, validation, testing을 한번씩만 split함
>- 그런데 만약에, 학습이 잘 되도록 training, validation 쪼개진것이라면?
>- 이것을 막기 위해 Cross Validation을 사용함





#### 교차검증

>vs. Validation Approach
>
>- 전체 데이터를 3개로 쪼개어 train, validation, test
>- train으로 fit
>- validation 으로 predict, 오차가 제일 낮은 모델 선택
>- test



- overfitting 을 방지하기 위하여 수행

- validation을 한번만 수행하면 특정 data에**만** 최적화될 수 있음(우연치 않게)

  - overfitting의 위험

- 다양하게 Training Data와 Validation Data를 변경하면서 모델평가를 수행

  예) 5개로 데이터셋을 쪼갬(1,2,3,4,5)

  |              |          |          |          |          |          |
| ------------ | -------- | -------- | -------- | -------- | -------- |
  | Experiment 1 | 오차측정 |          |          |          |          |
| Experiment 2 |          | 오차측정 |          |          |          |
  | Experiment 3 |          |          | 오차측정 |          |          |
| Experiment 4 |          |          |          | 오차측정 |          |
  | Experiment 5 |          |          |          |          | 오차측정 |

  - MSE1 ~ MSE5 ... 다섯개 에러의 평균을 실제 에러라고 생각함
- 전체 데이터에 validation이 걸쳐있기 때문에 "Cross Validation"이라고 한다



#### K-Fold Cross Validation

- Training Data를 무작위로 균등하게 K개의 그룹으로 나누어서 검증
  - (K-1)개의 Training Fold와 1개의 Validation Fold를 지정
  - K는 **Hyperparameter**
  - 일반적으로 K값은 5~10개 정도로 선택
  - K개의 **결과의 평균**을 Validation Data에 적용하여 평가
- Data가 충분히 많다면 K-Fold Cross Validation 수행
- 데이터가 매우 적다면 **데이터의 개수만큼 교차검증**을 수행(LOOCV)



(교재는 5.1.2까지 참조하면 cv사용엔 크게 지장이 없을 것)







## Gradient Boosting

(p.122)





#### 부스팅이란?

- 랜덤 포레스트는, 병렬처리가 가능
  - 나무 여러개를 한꺼번에 만드는 것이 가능
  - n_estimator = 3 이라면 나무 3개(=bootstrap data)가 동시에 만들어짐
  - 의사결정나무보다는 느리겠지만, 병렬로 속도가 빠른 편
  - 어느 bootstrap data가 좋을지는 크게 중요하지 않음
- 부스팅은, 순차적 처리
  - 원본데이터에서 Bootstrap Data를 뽑음(단순복원추출), 나무가 만들어짐
  - 나무 하나를 만든 후 성능을 측정(성능평가)
  - 성능평가한 것을 기준으로, 에러를 보완하는 방식으로 Bootstrap Data2 를 뽑아냄
    - 어느정도 오차를 반영할건지를 학습율(Learning rate)
  - 2를 기준으로 보완하는 방식으로 Bootstrap Data3
  - 순차적으로 처리되면서 반복학습. 일반적으로 나무를 많이 주면 성능이 좋아진다
  - 손실함수를 정의하고 경사하강법을 사용하여 다음에 추가될 트리가 예측해야 할 값을 보정해 나감(p.122) - GBM(Gradient Boosting Machine)





### 에이다부스트

(p. 131)

- 이전 모델이 잘못 분류한 샘플에 가중치를 높여서 다음 모델을 훈련



### XGBoost

### LightGBM





랜덤포레스트에서 교차검증 확인하기

cross validation score







## 리지회귀와 라쏘회귀



리지 회귀 (p.78)

라쏘 회귀(p.82)



### Overfitting



수치예측(Regression, Linear Regression)

- Model: y = wx + b

- 목적: 학습을 시켜 최적의 w, b를 찾는 것
  - w = w - r*dw
  - b = b - r*db
- 학습의 원리: 주어진 학습 데이터의 오차를 줄이는 것이 목적
  - Gradient Descent
  - OLR: Ordinary Linear Regression
  - LSM: Least Square Method

- Validation: MSE -> mean(y - y_hat)**2 을 최소화
- 모델링의 목적
  1. 과거 데이터를 잘 설명하는가?
  2. 미래 결과를 잘 예측하는가?
     - (Train_MSE ~ Test_MSE) Train과 Test의 갭이 작았으면 좋겠다!



- 과거데이터에만 잘 맞는 것을 막기 위해 규제를 준다
- 과거데이터를 너무 잘 학습하니, 학습을 좀 방해해보자! 는 개념



### Regularization (규제화)

> - 모델이 학습데이터에 너무 학습되지 않도록 방해하는 것 (Overfitting 회피)
>
> - 모델의 capacity에 영향을 주는 것이 파라미터의 개수
>   - 다항회귀
>   - 다중회귀



#### Overfitting 발생 원인

1. 데이터 포인트의 개수가 적을 때
2. Model Capacity가 높을 때



- Model Capacity에 규제를 줌

  (parameter를 구성하는 w의 값을 축소하겠다)

- overfitting이 줄어드는 형태로 동작하게 됨





예) 똑같은 데이터셋 3개가 있다고 하자

3개의 모델을 만듦 (각각 다르게)

1. 일차원 함수 (y = wx + b)

2. 이차원 함수 (y = w1x^2 + w2x + b)

3. 5차원 함수  (y = w1x^5 + w2x^4 + w3x^3 + w4x^2 + w5x + b)

   

- 어느 것이 제일 좋은 모델인가? 알수없다. Test를 해봐야 안다
- 과거 Train 데이터에 대해 너무 학습을 하며 오히려 Test 데이터에 대한 오차는 떨어질 수 있다 (Overfitting)



Train Data에 대해서는 1, 2, 5 .. 차수가 올라갈수록 점점 MSE가 내려가지만,

Test Data의 경우에는 내려가다가 어느 단계를 넘어서면 다시 MSE가 올라가게 된다

Train, Test 모두 오차가 내려가는 구간: Under fitting

Test 오차가 올라가는 구간: Overfitting

둘다 낮은 지점: 최적



5차에서는 너무 학습되므로, 해결책? 차수를 낮추면 되는 것으로 보인다

(y = w1x^5 + w2x^4 + w3x^3 + w4x^2 + w5x + b) 에서

(y = w4x^2 + w5x + b) 가 되면 차수가 낮아진다  (즉 w1, w2, w3이 0이 되면 된다)

"w의 값을 축소시키면 된다" w에 제한을 건다, 규제를 건다



Regularization에서는 학습의 식을 바꿔줘야 한다



MSE이든, CEE이든 학습의 원리에 따라 오차를 최소화하는데, train, test가 모두 낮은 점을 찾도록 해준다 







수치예측 오차함수(MSE)

mean(y - y_hat)**2

- 이 안에 w, b가 있다 (y = w1x^5 + w2x^4 + w3x^3 + w4x^2 + w5x + b) 

이 값을 최소화해야 한다 - Regularization



이진분류 오차함수(CEE)

sum(-y * log(y_hat) - (1-y)*log(1-y_hat))

- 이 안에도 w, b가 있다





최소화하는 것을 방해하기 위해서는 인위적으로 어떤 값을 더해준다

값을 더하면 전체적인 에러는 커진다

w값의 제곱의 합을 더해준다

- w가 마이너스일 수도 있으므로
- w의 값과 관계가 되어 에러가 증가함
- 최소값을 줄이기 위해서는 더해준 제곱의 합의 값이 줄어들어야 하고, 그러려면 w1, w2, .. 의 값이 줄어들어야 함
- 제곱해서 더해주는 것을 리지(Ridge, L2), 절대값을 더해주는 것을 라쏘(Lasso, L1)라고 한다
- 엘라스틱: 절대값도 더하고, 제곱도 더하고, 알파값으로 조정



- 제곱하면 값이 너무 커질 수 있으므로, 규제를 제어하는 알파값(일반적으로 0~1사이)을 곱해줘서 규제의 강도를 제어

  - 경사하강법의 learning rate처럼
  - 알파값이 0이면 규제가 없는 것, 1에 가까우면 규제가 강해짐
  - 규제가 강할수록 그래프가 축소되고 (2차원, 1차원까지 떨어질 수도 있다) 너무 규제되면 상수항까지 떨어질 수도 있다

  

- 무조건 성능이 좋아지는 것은 아니다

- 하이퍼파라미터 같은 개념









sklearn의 logistic regression은 기본적으로 L1, L2를 포함하고 있다





모델링을 했을 때, 오차를 어느 수준까지 줄여야 허용 가능한가?

추정하기 위한 어느정도의 근거는 있어야 한다



ex) bike demand

근거: 하루 평균 대여량

- 하루평균 대여량이 10대인데 오차가 40대?

실제 자전거 보유량

우리의 오차 수치가 비즈니스 상 수치를 보았을 때 수용 가능한가?



우리가 만든 모델이 비즈니스적으로 타당한지도 고려해야 함

(이것이 프로젝트)





날씨별, 온도별 차이가 컸다면?

날씨별, 온도별로 데이터프레임을 나눠서 오차를 재는 것이 더 정확하다

그래야 제한된 자전거 개수로 비즈니스를 최대한 운영할 수 있을 것이므로



알고리즘은 정해져있지만 어떻게 적용하고 활용하는지는 비즈니스에 대한 지식이 필요하다



모든 분석의 끝은? 스토리텔링... (납득이 되어야한다)





알파, C 규제의 강도 (반대)

헷갈린다... 잘 볼것

p. 88









## K-Nearest Neighbors

> - 분류, 예측 모두 사용 가능
> - 머신러닝의 "K"는 하이퍼파라미터(사용자가 조정), 동시에 학습대상
> - 최근 잘 사용하지는 않는다



K=5개의 이웃의 투표 결과에 따라 붉은색으로 분류





machine: 함수, 나무 외 다른 알고리즘들도 있다



머신러닝 지도학습 알고리즘 중의 하나

이웃한 데이터포인트를 바탕으로 하는 알고리즘



#### 데이터 분류시 이웃한 데이터포인트의 분류를 바탕으로 하는 알고리즘

- K-NN에서 K는 투표 과정에 참여할 최근접 이웃의 개수 파라미터
- 최적의 K 값을 찾기 ㅜ이하여 파라미터를 튜닝(교차 검증)

- 사기 탐지(Fraud Detection), 이상감지에 적합



#### 데이터 시각화를 통하여 용이하게 분류 가능

- 다차원 공간에서 계산량이 급격히 증가
- 예측자의 개수가 적고, 분류의 크기가 비슷할 때 권장



최근접이웃 (p.63)





---

### 지도학습 총정리

- AI라는 것은, 사람의 지적작업을 대신해줄 수 있는 무언가를 만드는 것
- 그것을 함수를 학습시켜 만들겠다는 것이 "3차 인공지능의 시대"
- 사람의 지적작업 중 예측작업, (수치예측, 분류예측)을 함수를 학습시켜 처리하는 것
- 함수형 모델 외에 나무형 모델이 있다
- 함수형이 확장되어 딥러닝의 NN(Neural Network)이 된다
- 신경망은, 함수와 함수를 연결시키는 것(연결시키면 다른 형태로 작업하더라)
- 함수의 파라미터나 나무의 룰을 조정해서 예측/분류작업을 수행
- "머신"은 함수, 혹은 나무로 보면 된다. 그에 따라 다양한 모델과 알고리즘이 나오게 된다
- 지도학습을 하기 때문에 평가가 가능
- 평가에는 대부분 오차를 본다(예측에서는 MSE, 분류에서는 CEE)
- 오차를 최소화하는 방향으로 학습
- 오차 측정은 정성적인 것이 아니라 정량적으로 측정한다
- 오차를 줄이기 위해서 자동화시키는 기법이 "Gradient Descent"
  - 딥러닝에서 보다 자세히 보게 될 것
- 함수형 모델의 경우 경사하강법을 통해 최적의 w,b를 찾아내게 됨
- 모델의 데이터에 대한 설명이 어느정도 잘 될것이냐? 를 validation함
- 데이터 분석의 목적
  - 과거 행동의 특징 확인(Train Data로 확인)
  - 미래 행동의 결과 예측(Validation, Test Data로 확인)
- 데이터를 나누는 방법
  - 데이터프레임으로 나누는 방법
  - array로 나누는 방법(일반적)
- Train/Test Data의 분포가 같아야 한다
- 미래의 행동을 잘 예측하기 위해서는 일반화된 모델을 만드는 것이 중요하고 그러기 위해서는 Overfitting되지 않은 모델이 필요하다
- overfitting: 학습용 데이터가 너무 적은가? model capacity가 너무 높은가?
- 학습데이터를 늘리거나 mdel capacity에 영향을 주어야 함(L1, L2)

- 정규화: normalization, standardization

- R-square: 모델의 데이터에 대한 설명력. 일반적으로는 MSE를 더 많이 쓴다

- 다중회귀가 되면 공간상의 면으로 모델이 만들어지고, 여전히 점과 면 사이의 거리를 최소화시키는 방향으로 모델링이 된다

- 분류: Sigmoid = Logistic 
  - 0~1 사이의 값으로 축소시킴
  - 기본적으로 이진분류, 다중분류도 가능하다(함수형으로 다중분류하는 방법은 딥러닝에서 더 자세히)
- 분류문제의 Validation: Accuracy, Precision, Recall

- 분류문제에서는 Cross Entropy Error
- 나무모델: Rule Set을 찾는 것
  - Rule 을 찾는 원리: Entropy를 낮추는 형태로

- 군집을 한다는 것은, "비슷하다는 느낌"(유사도)
  - 비슷한 정도가 높은 것들끼리 묶는 것
  - 비슷하다는 기준은 무엇일까? 
  - 답은 없지만 기준은 가져가야 한다
- 의사결정나무의 문제점: Overfitting이 일어날 수 있다
- 해결책: Pruning
- Random Forest는 나무를 병렬로 만듦
- Boosting은 나무를 순차적으로 만듦(오답노트 방식으로 나무를 만들어 감)



---



- 시계열데이터는 처리방법이 다르다

  ex) 여름의 데이터로 학습해 겨울의 데이터를 테스트한다...







## 8. Association Rules(연관규칙)

> Machine Learning
>
> - Supervised Learning
>
>   - Regression
>
>   - Classification
>
> - Unsupervised learning: Group and interpret data based on **only input data**
>
>   - Clustering (군집, 연관)
>   - 차원축소 (=특징 추출): 딥러닝에서 살펴볼 것



- 데이터간의 연관된 규칙을 찾아내겠다

- 맥주를 산 사람이 감자칩을 사더라

- 감자칩을 구매한 고객은 맥주때문에 감자칩을 구매했느냐?를 찾고싶은 것

- 마트, 전자상거래(옥션, 지마켓 등)에서 "이 물건을 산 사람은 이 물건을 샀다"

  - 연관규칙 기반의 추천시스템

- 처음 전자상거래에 도입한 업체: 아마존

- 롱테일

  - 온라인, 오프라인 서점이 함께 존재하던 시절
  - 온라인, 오프라인 가격경쟁이 아니라 온라인-온라인 가격경쟁을 하게 되면 마진이 줄어들게 되고, 치킨게임의 형태가 됨
  - 아마존이 포커싱한 것은, 오프라인에서 팔지 않는 책을 판매하는 것(롱테일 법칙)
  - 오프라인 서점에는 잘 팔리는 책만 둘 수밖에 없다
    - 인기서적 20%만 매장에 둘 수밖에 없다
  - 오프라인 매장에서 살 수 없는 80%의 책을 소비자와 가상으로 연결시켜주자
  - 상위 20% 매출이나 80%의 매출이나 매출액은 비슷하다
  - 연관규칙으로 추천을 해주며 온라인 서점으로 급성장하게 됨
  - 책 이후 추가로 음반으로 사업을 확장함

  

- Netflix도 처음엔 비디오대여점(비디오, DVD)이었다

- 마케팅, 추천시스템에 관심이 있다면 연관규칙이 유사도를 측정하는 방법 중 하나이다

- 다른 추천시스템과 차별화된 점: 방향성이 존재







- 지도학습은 y가 존재하므로 모델을 만들어낼 수 있었다
- X와 y의 관계를 학습(=Data의 모양을 근사)





### 비지도 학습

- 데이터에 존재하는 특징을 확인(군집, 연관)

  - 특징이 비슷한(유사한) Data Point로 묶거나(군집), 관계를 찾아내려고 하는 것(연관)

- Input(x) Data만 제공(답이 존재하지 않음)

  - 평가기준: **Business(업무)**, 정량적인 기준을 찾기 어려움. 정성적

    ex) 소비자를 그룹으로 묶고 싶은데, 몇개의 그룹으로 묶는 것이 좋을까? 비즈니스에서 선택 (2개? 5개? 8개?)

    수학적 값을 통해 최적의 값을 recommendation할 수는 있으나, 그것이 최적인지 분석가는 알 수 없다

- 일반적으로 지도학습을 위한 **전처리 작업**으로 수행







### Association Rules

> 어떤 사건들 간의 관계성을 확률로 계산 
>
> 기본적 원리: 조건부확률
>
> 데이터셋을 축소하기 위해 Support 라는 개념
>
> 

#### 연관규칙



#### 데이터 포인트 사이의 관계된 규칙을 찾는 방법

- 규칙(Rule) - "If 조건 then 결과" 형식



#### 연관규칙: 특정사건 발생 시 함께 자주 발생하는 다른 사건의 규칙(Rule)

- 특정사건 발생 시 함께 자주 발생: "조건부확률"
- 확률: "특정 비율로 굳어진다", 전체사건 중 특정사건이 발생할 비율 

- 상품A를 구매한 고객이 상품B를 **함께 살** 확률(구매패턴)을 확인
- 마케팅에서 고객 별 장바구니 내의 품목 간의 관계를 분석
  - 이 물품 때문에 저 물품이 함께 구매가 되었는가?
- 타겟 마케팅, 1)효과적 매장 진열, 2)패키지 및 신상품 개발에 활용
  - 1) A를 사는 사람이 B도 반드시 산다면, 두 상품을 멀리 떨어뜨려놓는 게 더 일반적
  - 2) 패키지: 어차피 살텐데 패키지로 묶어서 팔 이유가 있을까?





확률을 계산하는 방식 세가지



| 고객 A | 감자칩, 맥주, 쥐포, 빵 |
| ------ | ---------------------- |
| 고객 B | 감자칩, 맥주, 쥐포     |
| 고객 C | 감자칩, 맥주           |
| 고객 D | 감자칩, 콜라           |
| 고객 E | 기저귀, 맥주, 쥐포, 빵 |
| 고객 F | 기저귀, 맥주, 쥐포     |
| 고객 G | 기저귀, 맥주           |
| 고객 H | 기저귀, 콜라           |



#### 지지도(Support)

> 지지도가 높게 나타나야 자주 등장한다는 의미
>
> 전체거래량이 너무 많으므로 원하는 품목을 고르기 위해 사용 (Threshold를 지정)
>
> 함께 빈번하게 나타나는 품목을 고르기 위해 사용 
>
> 지지도로 뽑아내어 confidence, lift 계산



- 특정품목 집합이 얼마나 **자주 등장**하는지 확인
- 특정품목 집합을 포함하는 거래의 비율로 계산

- {감자칩}은 8번 중 4번 = 지지도 50%
- {감자칩, 맥주, 쥐포}는 8번 중 2번 = 지지도 25%
- 지지도 임계값(Support Threshold)을 **빈번한 품목 집합을 구분**하는 기준
  - 임계값보다 지지도가 큰 품목은 빈번히 등장하는 것으로 볼 수 있음



- 전체 거래에 대한 A와 B가 동시에 발생할 확률
- Support = (A와 B가 동시에 포함된 거래수) / 전체거래수
- Support(A->B)와 Support(B->A)는 같은 값을 가짐: 방향성 없음
- Support({})







#### 신뢰도(Confidence): {상품A} -> {상품B}

> 방향성 존재

- 상품A가 존재할 때 상품B가 나타나는 빈도

- 상품A가 포함된 거래 중 상품B를 포함한 거래 비율

- 원인과 결과가 존재

  - {감자칩}(원인) -> {맥주}(결과) 네 번 구매 중 세 번 발생 = 75%

    

- 그러나 감자칩을 구매했다고 맥주를 구매했다고 할 수 없다. 왜?

- 감자칩은 4번, 맥주는 6번... 감자칩이 아니어도 맥주는 사람들이 많이 사는 것일 수 있다

  ex) 사람들이 대표적으로 많이 사는 생수. 일반적으로 생수의 confidence는 높다

  

- 맥주의 판매 빈도는 고려하지 않고 감자칩의 판매 빈도만 고려

  - 맥주 자체가 자주 거래되는 상품이라면 신뢰도를 부풀릴 수 있음

- 조건부확률 만으로 설명할 수 없다



- 향상도(Lift)를 활용하여 두 물품 모두의 기반 빈도(Base Frequency)를 고려



- 조건 발생 시 결과가 동시에 일어날 확률
- A가 구매될 때 B가 구매되는 경우의 조건부 확률
- Confidence = (A와 B가 동시에 포함된 거래수) / A를 포함한 거래수
- Confidence({감자칩} -> {맥주}) = Support(A, B) / Support(A) = 0.75





#### 향상도(Lift) {상품A} -> {상품B}

> 방향성 있음

- 두 물품이 각각 얼마나 자주 거래되는지를 고려
- 상품A와 상품B가 함께 팔리는 빈도
- Confidence{상품A} -> {상품B}를 상품B의 빈도로 나눈 것
- 향상도{상품A} -> {상품B} = **1** (1은 두 물품 사이에 연관성이 없음. 독립사건)
  - 향상도가 1보다 크면 상품A 거래 시 상품B도 함께 거래될 가능성 있음
    - 상품A 때문에 상품B가 팔렸다(연관이 생김)
  - 향상도가 1보다 작으면 상품A 거래 시 상품B가 거래될 가능성 작음
    - 대체재 ex) 박카스와 비타500
- Lift = Confidence({A} -> {B}) / Support({B})
- Lift({감자칩} -> {맥주}) = Confidence({감자칩} -> {맥주}) / Support({맥주})

- 감자칩 때문에 맥주를 샀다고 할 수 없다. (독립관계)









## 9. K-means Clustering

> - K: 군집을 몇개로 할지 지정



- K 개수만큼 랜덤 점을 찍어 clustering

- 묶인 그룹의 평균 지점으로 이동, 다시 clustering
- 다시 묶인 그룹의 평균지점으로 이동, 다시 clustering
- 점이 움직이지 않을 때까지 반복

- 더이상 움직이지 않으면 convergence 되었다고 함 (학습이 완료)
- 몇개로 묶는 게 좋은지는 비즈니스 의사결정



- 같은 그룹 안 점들 사이의 거리는 가까울수록 좋다 (밀도가 적고 분산이 작음)
- 그룹 간의 거리는 멀수록 좋 (그룹 간 차별성이 있어야 하므로)
- 비즈니스적으로 의사결정하기 어려울 때는 수학적 recommendation은 줄 수 있다
  - 그룹 내 거리는 가깝고, 그룹 간 거리는 먼 것





p.225



- 입력값에 대한 출력밧이 정해져 있지 않은 비지도학습 분석법
- 의사 중심점(Pseudo Center, =K)을 각 군집 내 평균점으로 
- 군집의 데이터 간의 포함관계는 어떤가?
- 데이터 간 1)유사성(Similarity)을 계산하여 유사성이 높은 개체의 군집을 생성
  - 1) 거리: (1)군집 내 거리, (2)군집 간 거리
- 여러 개의 군집(Cluster)을 생성하여 입력값이 속하는 그룹을 지정
- 동일한 그룹에 속하는 데이터는 유사성이 **높고**, 그룹 간에는 유사성이 **낮음**
  - 타킷 마케팅 캠페인을 위한 구매 패턴 그룹 세분화
  - 카드 부정사용, 불법 네트워크 침입 등의 이상 행동 탐지
  - 유사한 값을 갖는 특징을 적은 개수의 동질적 그룹으로 단순화



- 입력된 데이터를 K개의 군집(Cluster)으로 묶는 분석방법
- 각 군집 내의 데이터들 간의 거리를 최소화
- 각 군집들 간의 거리를 최대화
- 각각의 데이터는 **오직 한 개의 군집에만** 포함됨
  - 여러개의 군집에 포함되는 것을 하기 위해서는 다른 군집분석(계층적 군집분석)을 사용해야 함
- 최초 K개의 의사중심점(Pseudo Center)을 지정
- 분류된 데이터들의 평균점을 구하고 이동하는 과정을 반복



- 입력되는 모든 데이터가 좌표 평면에 표현될 수 있는 숫자데이터여야 함
  - 예) 오렌지-바나나, 오렌지-귤 의 거리는?
  - 자연어 처리에서도 단어를 벡터로 만드는 word3vec 같은 방식이 사용된다
- 몇 개의 군집으로 분류할 것인가?
  - 주관적으로 비즈니스 의사결정에 도움을 주는 수를 권장
  - 군집의 개수를 늘리면 데이터 간 유사성 증가, 인접 군집과 차이점 감소
  - 군집 내 거리와 군집 간 거리를 수치화 해 metric화 하는 식이 있음
- 스크리 도표(Scree Plot) 활용
  - 킨크(Kink): 스크린 도표가 급격히 구부러진 부분(군집 내 산포도가 적당한 수준으로 떨어지는 지점)





- from sklearn.cluster import KMeans
  - n_clusters: 군집개수 지정
  - init: 초기 중심 설정 방식 (초기 찍히는 점에 따라 결과가 달라지는 것을 방식)
  - max_iter: 최대 반복 횟수
  - fit: 데이터 포인트를 n_clusters 개수만큼 묶음





### 군집분석 성능평가: Silhouette Analysis

- 수학적인 관점에서 군집분석의 성능이 좋다는 것은,
  1. 군집 내의 데이터포인트들의 거리는 가깝고,
  2. 군집 간의 데이터포인트들의 거리는 멀다
- Cluster A, Cluster B, Cluster C
  - A의 관점에서 C보다 B가 가까울 때, (B가 최근점) 
  - Cluster A 안에 1, 2, 3, 4
  - Cluster B 안에 5, 6, 7, 8
- 거리 계산: 모든 데이터포인트들의 거리 계산
  - A(i): ex) A(12), A(13), A(14)... 같은 군집 내 다른 데이터포인트 거리의 평균
    - 군집 내의 거리 표현
  - B(i): ex) B(15), B(16), B(17), B(18)... 1이 포함되지 않은 군집 중 가장 가까운 군집 내의 데이터포인트와의 거리의 평균
    - 군집 간의 거리 표현
  - A와 B값을 가지고 Silhouette Score(Coefficient)를 계산할 수 있다
  - S(i) = B(i) - A(i) / Max(A(i) , B(i)): B에서 A를 뺀 값을 A(i)와 B(i)중 큰 값으로 나눔(보통 B)
  - Silhouette Coefficient를 평균낸 것이 Silhouette Score
  - 데이터마다 다 계산해서 평균낸 것이 실루엣 계수. 1에 가까울수록 좋은 것

- 수학적인 계산이므로, 정말로 좋은 것인지 장담할 수는 없다





### Mean Shift

- 데이터 포인트별로 확률밀도함수가 존재한다고 가정
- 각각을 합쳐서 다시 밀도함수를 그림
- bandwidth(대역폭)가 1에 가까우면 좁은 형태의 밀도함수를 가지게 됨
- bandwidth가 10.0이면 분산이 큰 형태의 밀도함수를 가지게 됨
- bandwidth에 따라 합쳐지는 개수가 달라짐
  - bandwidth를 좁게 설정하면 군집의 개수가 많아짐
  - bandwidth를 넓게 설정하면 군집의 개수가 적어짐



- bandwidth: 군집으로 묶일 분포의 분산: 넓게 볼건지, 좁게 볼건지 (통계학의 첨도)
- 새로운 분포를 만드는데 분산을 어느정도로 가져갈지 

### GMM

- 정규분포: 평균을 중심으로 좌우 대칭인 분포
- 정규분포를 찾아내는 과정
- 군집의 수를 정해주는 알고리즘이 있고, 기준을 주면 군집해주는 알고리즘이 있다





### DBSCAN(Density Based Spatial Clustering of Application with Noise)

- 밀도(Density) 기반 군집

  - 기하학적으로 복잡한 데이터에도 효과적으로 군집 가능
  - 핵심 포인트(Core Point)들을 서로 연결하면서 군집화
- ex) 데이터포인트 하나가 있고, 그 주변에 다른 포인트들이 있다고 가정
  - 특정 점(핵심포인트)에서 1) 입실론 반경을 지정하고 원을 그림
    - 1) 입실론: 원의 반지름
  - min_samples = 5: 입실론 내에서 5개 이상의 데이터포인트를 가지고 있으면 core point가 된다
  - core point와 core point를 군집화하는 방식
  - esp와 min_samples로 밀도를 조정, 밀도가 높은 데이터끼리 군집화해나감
  - metric = 'euclidean' : 직선으로 가는 거리(유클리디안), 직각으로 재는 거리(맨하탄)



무엇이 가장 좋은지는, 비즈니스에 따른 의사결정













p. 235 PCA로 얼굴의 특징만 뽑아냄







## Principal Component Analysis (주성분 분석)

> 차원이 높은 데이터의 차원을 낮춰주는 방법
>
> 29개 변수... 너무 많다, 대표하는 10개로 차원축소하여 분석한다! 는 개념
>
> 딥러닝에서 다시 볼 것 (GAN 알고리즘으로 가는 길목, auto encoder)



- 각 항목을 잘 구별해주는 변수를 찾는 일
  - 데이터포인트를 가장 잘 구별해주는 배후의 변수(주성분)를 찾는 기법
    - 많은 수의 x를 적은 수의 x로 축소시킨다고하여 차원 축소 기법
  - 주성분으로 데이터 세트를 표현 가능(차원축소기법)
  - 두 개의 변수가 있을 때(x1, x2), 특징을 다 가지고 있는 새로운 형태의 변수(목적변수)를 만들어냄
  - "축을 돌린다"고 표현. 분산이 큰 지점을 찾아냄
    - ex) 영화에 대해 IPTV 시청수, 극장 관객수를 알고있다면, x축 y축으로 표시
    - 분산이 제일 큰 지점에 선을 긋고, 선을 새로운 변수로 지정 (영화 인기도)







일반적으로 많이 사용되는 군집분석의 내용





