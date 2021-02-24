# R 기본 함수 #### 

print("Hello World")

"Hello World"





# 에러발생

print("Hello", "World")



# 여러 단어 출력

cat("Hello", "World")






# 에러발생 - 미지정 객체(Object)

print(koo)



# 객체값 할당

koo <- "Hello World"



# 객체값 출력

print(koo)


koo







# 문자, 내장객체, 사용자지정객체 출력

cat("Hello World", pi, koo)



# 여러 함수 실행

print("Hello World") ; print(pi) ; print(koo)



# 이탈문자(\n, \t) 사용

cat("Hello World", "\n", pi, "\t", koo)





# 객체명 지정 규칙 ####
# 1) 객체명은 숫자그리고 특수문자로 시작 불가
# 2) 객체명에 사용 가능한 특수 문자 : ".", "_"
# 3) 객체명에 공백문자(" ") 포함 불가
# 4) 객체명은 대소문자 구별

# Qoo1에 9 할당

Qoo1 <- 9



# Qoo2에 3 할당

Qoo2 <- 3



# Qoo2의 값(3)을 Qoo1에 할당

Qoo1 <- Qoo2



# Qoo2에 7 할당

Qoo2 <- 7



# Qoo1 vs Qoo2

Qoo1

Qoo2





# 지정된 객체 확인

objects()


rm(Qoo2)
Qoo2



ls()

rm(list = ls())

ls()





# Data Type ####

num <- 999
class(num)


char <- "R Program"
class(char)


logi <- TRUE
class(logi)


com <- 2 + 8i
class(com)




# 숫자형

1e+1

1e+2

1e+3



1e-1

1e-2

1e-3





# 산술연산 ####

10 + 3

10 - 3

10 * 3



10 / 3

10 %/% 3

10 %% 3


10 ^ 2



# 연산 우선순위

7 + 8 * 9

(7 + 8) * 9




# 반올림연산

round(24.47)


round(24.56)

round(24.56, 0)


round(24.99, -1)

round(25.51, -1)

round(24.35, 1)



# 올림

ceiling(24.51)



# 내림

floor(24.51)



# 반올림, 올림, 내림, 버림

koo <- c(3.54, -3.14, 2.14, -2.54)

round(koo)

ceiling(koo)

floor(koo)

trunc(koo)









# 에러발생

"8" + "9"


# 타입변환

as.numeric("8") + as.numeric("9")

class(8)

class("8")


# 데이터 타입 변환 함수
# as.list()
# as.vector()
# as.factor()
# as.integer()
# as.numeric()
# as.character()
# as.data.frame()



# 주의

as.integer(3.14)
as.integer(-3.14)

# 데이터 타입 판별 함수
# is.list()
# is.vector()
# is.factor()
# is.integer()
# is.numeric()
# is.character()
# is.data.frame()












# R내장 수학함수
# sin(), cos(), tan()

sin(45)
cos(45)
tan(45)




# root

sqrt(8)
sqrt(16)






# log()

log(100, base = 10)
log(100, base = 5)
log(100)





# 절대값

abs(-3.14)
abs(-2.14)






# factorial()

factorial(5)
factorial(7)
factorial(9)





# Combination(조합)

choose(5, 2)
choose(7, 3)






# 논리연산 ####

88 > 99
88 < 99

"오이" == "호박"
"오이" != "호박"

88 >= 99
88 <= 99



# 주의

"팔" > 9




# AND/OR

x <- TRUE ; y <- FALSE

x & y
x | y

!x
isTRUE(x)

z <- TRUE

(x | y) & (z & y)







# 벡터 ####
# R의 최소 데이터 단위

a <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
is.vector(a)

b <- c(1, 3, 5, 7, 9)
is.vector(b)

c <- c(9)
is.vector(c)


d <- c(1, "삼", 5, "칠", 9)
is.vector(d)
class(d)




# 벡터 길이 확인

koo <- c(2, 4, 6, 8, 10, 12, 14, 16, 18)

length(koo)



# 특정 문자 포함 여부 확인

8 %in% koo
9 %in% koo








# seq() 함수 사용하기

a <- seq(1, 10)

b <- seq(5, -5)

c <- seq(1, 10, 2)

d <- seq(from = 1, to = 10, by = 2)

e <- seq(from = 1, to = 9, by = 0.5)

f <- seq(from = 1, to = 9, by = 1.2)

1:10

1.5:5

1.5:6

9:1

10:-10

1:length(a)





# rep() 함수 사용하기

koo <- rep(3:5, times = 2)

koo <- rep(3:5, each = 2)

rep(1:4, 1:4)

rep(1:4, c(3, 1, 5, 2))

rep(c("HPE", "MS", "EMC", "IBM"), c(5, 1, 3, 2))









# sample( ) 함수

qoo <- sample(1:45, 6)

qoo <- sample(1:45, 7)

qoo <- sample(1:6, 3, replace = TRUE)



# runif( ) 함수

koo <- runif(10)

koo <- runif(10, min = 1, max = 100)

koo <- round(runif(10, 1, 100))













# 벡터 산술연산
# 신체질량지수 계산

Height <- c(163, 177, 156, 183, 188)
Weight <- c(69, 80, 43, 78, 95)

BMI <- Weight/(Height/100)^2



# 벡터의 재사용

p <- 1:10
q <- c(1, -1)

p + q














# 벡터 논리연산

lee <- 1:15

lee < 8

1 * (lee < 8)



# 벡터 정렬

lee <- sample(9:99, 10)

sort(lee)

rev(lee)

rev(sort(lee))

sort(lee, decreasing = TRUE)

rev(sort(lee))[3:7]

rank(lee)



# 함수 적용

sum(lee)

sum(sort(lee)[3:7])

mean(lee)

median(lee)

var(lee)

sd(lee)

range(lee)

quantile(lee)





# 최대/최소 및 위치확인

max(lee)

min(lee)

which.max(lee)

which.min(lee)

which(lee > 55)

lee[lee > 55]







# 결측치(NA : Not Available)

qoo <- seq(1, 30, 3)

is.na(qoo)

qoo[c(4, 8)] <- NA

is.na(qoo)

sum(qoo)

sum(qoo, na.rm = TRUE)

sum(is.na(qoo))

qoo[qoo > 8]



# 결측치 제거

qoo.omit <- na.omit(qoo)

sum(is.na(qoo.omit))

sum(qoo.omit)

qoo.omit[qoo.omit > 8]



# 결측치 정렬

sort(qoo)

sort(qoo, na.last = TRUE)

sort(qoo, na.last = FALSE)

rank(qoo)

rank(qoo, na.last = TRUE)

rank(qoo, na.last = FALSE)





# 벡터 이름속성 지정

cars <- c(24000, 17000, 23000)

names(cars) <- c("Renault", "Picasso", "Peugeot")

cars

cars["Picasso"]

attributes(cars)

attributes(cars) <- NULL

cars















# Factor ####

# Vector 종류

v1 <- c(1, 3, 5, 7, 9)
is(v1)

# v1 <- c(1L, 2L, 3L)
v1 <- as.integer(v1)
is(v1)

v2 <- c(3.14, 2.34, 4.56789)
is(v2)

v3 <- sample(c("남자", "여자"), 10, replace = TRUE)
is(v3)




# factor() 사용하여 명목형, 순서형 지정하기

# 명목형 - levels 순서 미지정

v4 <- as.factor(v3)
is(v4)
nlevels(v4)
levels(v4)


# 명목형 - levels 순서 미지정

v5 <- factor(sample(c("금메달", "은메달", "동메달"),
                    15, replace = TRUE))

is(v5)




# 명목형 - levels 순서 지정

v6 <- factor(v5, 
             levels=c("동메달", "은메달", "금메달"))

is(v6)




# 명목형 - order 지정

v7 <- factor(v6, ordered = TRUE)
is(v7)
nlevels(v7)
levels(v7)




# 데이터프레임 ####

# 여러개의 벡터로 데이터프레임 만들기

v1 <- c("Renault", "Picasso", "Peugeot", "Focus", "Fiesta")
v2 <- c(24000, 17000, 23000, 15000, 12000)
v3 <- c(3, 5, 2, 2, 3)



DF1 <- data.frame(v1, v2, v3)
DF1


DF1 <- data.frame(Model = v1,
                  Price = v2,
                  Count = v3)



# 데이터프레임 사용하기

DF1$Model

DF1[3, 1]

DF1[1, ]

DF1[ , 3]

DF1[c(3, 5), ]

DF1[ , c(1, 3)]

DF1[2:4, ]

DF1[ , 1:3]



# subset() 함수 사용하기

subset(DF1, Count > 2)

subset(DF1, Model == "Focus")

subset(DF1, Price >= 20000)

subset(DF1, Count > 2 & Price > 20000)

subset(DF1, Price < 20000 | Count > 2)

subset(DF1, Price < 19000 & Count == 2)







# cbind(), rbind() 함수 사용하기

No <- c(1, 2, 3)
Model <- c("Renault", "Picasso", "Peugeot")
Price <- c(24000, 17000, 23000)

DF1 <- data.frame(NO = No, 
                  MODEL = Model, 
                  PRICE = Price)



No <- c(101, 102, 103)
Model <- c("HPE", "IBM", "ORACLE")
Price <- c(9000, 8000, 7000)

DF2 <- data.frame(NO = No, 
                  MODEL = Model, 
                  PRICE = Price)



cbind(DF1, DF2)

rbind(DF1, DF2)





# merge() 함수 사용하기

DF1 <- data.frame(Server = c("HPE", 
                             "IBM", 
                             "ORACLE"), 
                  Memory = c(4096, 
                             2048, 
                             1024))


DF2 <- data.frame(Server = c("HPE", 
                             "DELL", 
                             "IBM"), 
                  CPU = c(64, 
                          32, 
                          16))



merge(DF1, DF2)

merge(DF1, DF2, all = TRUE)



cbind(DF1, DF2)



# 에러발생

rbind(DF1, DF2)








# rbind()로 행추가

New <- data.frame(Server = "Dell",
                  Memory = 2048)

DF1 <- rbind(DF1, New)

DF1 <- rbind(DF1, 
             data.frame(Server = "EMC", 
                        Memory = 4096))



# cbind()로 열추가

DF1 <- cbind(DF1, 
             data.frame(CPU = c(64,
                                32,
                                16,
                                32,
                                64)))









# 데이터프레임 실습

no <- c(1, 2, 3, 4, 5)

name <- c("유재석", 
          "박명수", 
          "정준하", 
          "양세형", 
          "조인성")

address <- c("서울", 
             "용인", 
             "창원", 
             "광주", 
             "부산")

tel <- c("02", 
         "031", 
         "055", 
         "062", 
         "051")

hobby <- c("농구", 
           "독서", 
           "영화", 
           "맛집", 
           "수영")

Member <- data.frame(NO = no,
                     NAME = name,
                     ADDRESS = address,
                     TEL = tel,
                     HOBBY = hobby)


subset(Member, 
       select = c(NO, NAME, TEL))

subset(Member, 
       select = -TEL)

colnames(Member) <- c("번호", 
                      "이름", 
                      "주소", 
                      "전화", 
                      "취미")

ncol(Member)

nrow(Member)

names(Member)

colnames(Member)

rownames(Member)

Member[c(1, 3, 5),]












# 리스트 ####

list_1 <- list(name = "Lee Na Young",
               email = "abc@abc.com",
               mobile = "010-1234-5678",
               height = 183)
list_1


list_1$name

list_1[2:4]

list_1$birth <- "12-25"

list_1$name <- c("Lee Na Young", "qoo")

list_1$birth <- NULL



# 리스트 실습용 벡터 생성

# 숫자형 벡터

v_i <- c(1:15)



# 문자형 벡터

v_c <- c("축구", 
         "야구", 
         "농구", 
         "배구", 
         "족구")



# 논리형 벡터

v_l <- c(F, T, F, F, T, T)



# 리스트, 데이터 프레임, 벡터들로
# 새로운 리스트 생성
# 이름 지정 안함

New_List <- list(list_1, Member, v_i, v_c, v_l)
New_List



# 실습용 리스트 생성
# 이름 지정

Test_List <- list(LIST = list_1, 
                  DataFrame = Member, 
                  Integer = v_i, 
                  Character = v_c, 
                  Logic = v_l)



# 리스트의 첫번째 값 확인

Test_List[1]



# 리스트의 첫번째 값 삭제

Test_List[1] <- NULL



# 두번째 값이 첫번째로 이동

Test_List[1]



# 리스트 이름으로 호출

Test_List["Integer"]



# 데이터 타입 확인

class(Test_List[2])
class(Test_List[[2]])



# 두번째, 네번째 값 확인

Test_List[c(2, 4)]



# 이름으로 호출

Test_List$Character



# 데이터 타입 확인

class(Test_List$Character)



# 리스트 이름 지정

names(Test_List)[2] <- "Number"



# 두번째 리스트 확인

Test_List[2]

















# 행렬(Matrix) ####
# 동작원리

matrix(1:16, nrow = 4)
matrix(1:16)
matrix(1:16, nrow = 2)
matrix(1:16, ncol = 2)



# byrow

matrix(1:16, nrow = 4, byrow = TRUE)
matrix(1:16, byrow = TRUE)
matrix(1:16, ncol = 16)
matrix(1:16, nrow = 4)




# 영문자

matrix(letters[1:16], nrow = 4)
matrix(LETTERS[1:16], nrow = 4, byrow = TRUE)



# 행이름, 열이름 지정

matrix(1:16, nrow = 4,
       dimnames = list(c("R1", "R2", "R3", "R4"),
                       c("C1", "C2", "C3", "C4")))



MX <- matrix(1:16, nrow = 4)
dimnames(MX) = list(c("R1", "R2", "R3", "R4"),
                    c("C1", "C2", "C3", "C4"))
MX



# 행이름 확인

rownames(MX)



# 열이름 확인

colnames(MX)



# 행렬값 접근

MX[1, 3]
MX[2, 4]
MX[2:3, ]
MX[ , c(2, 4)]
MX[-2, ]
MX[c(1, 3), c(-2, -4)]



# 연산을 위한 행렬 생성

MX1 <- matrix(1:16, nrow = 4)

MX2 <- matrix(16:1, nrow = 4)



# 행렬과 스칼라 연산

MX1 + 3
MX2 - 3
MX1 * 3
MX2 / 3



# 행렬과 행렬 연산

MX1 + MX2
MX1 - MX2
MX1 %*% MX2



# 행렬의 차원

dim(MX1)







# 배열(Array) ####
# 동작원리

array(1:16)
array(1:16, dim = c(4, 4))
array(1:16, dim = c(2, 2, 4))




# 배열값 접근

AR <- array(1:16, dim = c(2, 2, 4))
AR[1, 2, 3]
AR[,, 4]





# 배열의 차원

dim(AR)







# The End ####