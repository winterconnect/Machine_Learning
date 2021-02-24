# 실습용 데이터프레임 생성 
 
DF <- read.csv("[readData]/PII.csv")







# *apply() 함수 개요 ####

# 데이터 전처리 함수
# 동일한 '함수'를 '반복' 적용

# 총 5개의 함수로 구성
# apply(데이터, 연산방향, 적용함수)
# lapply(데이터, 적용함수)
# sapply(데이터, 적용함수)
# mapply(적용함수, 데이터)
# tapply(데이터, 인덱스, 적용함수)


# 그룹별 연산
# aggregate(종속변수 ~ 독립변수, 데이터, 적용함수)
# by()




# apply(데이터, 연산방향, 적용함수) ####
# 연산방향 : 1 = 행기준, 2 = 열기준

DF.aply <- data.frame(DF$Height, DF$Weight)

# 상위 6행 출력

head(DF.aply)


# (예제) 행기준 연산

apply(DF.aply, 1, sum)

apply(DF.aply, 1, mean)



# (예제) 열기준 연산

apply(DF.aply, 2, sum)

apply(DF.aply, 2, mean)







# lapply(데이터, 적용함수) ####
# l(영문자 엘)
# 벡터, 데이터프레임, 리스트 입력가능
# (예제) 리스트로 결과 반환

lapply(DF.aply, sum)


lapply(DF.aply, mean)


lapply(DF.aply, median)








# sapply(데이터, 적용함수) 적용 ####
# 벡터, 데이터프레임, 리스트 입력가능
# (예제) 벡터로 결과 반환

sapply(DF.aply, sum)


sapply(DF.aply, mean)


sapply(DF.aply, median)









# mapply(적용함수, 데이터) ####
# multivariate
# 다수의 벡터를 적용함수에 전달

# 적용함수의 위치 확인

mapply(sum, DF[, 7:8])

sapply(DF[, 7:8], sum)



# mapply 두개의 벡터 전달

mapply(sum, DF$Height, DF$Weight)



# sapply 두개의 벡터 전달 - 에러발생

sapply(DF$Height, DF$Weight, sum)



# 데이터프레임으로 전달 - 정상처리

sapply(DF[,7:8], sum)



# apply 두개의 벡터 전달- 에러발생

apply(DF$Height, DF$Weight, 1, sum)



# 데이터프레임으로 전달 - 정상처리

apply(DF.aply, 1, sum)











# tapply(데이터, 인덱스, 적용함수) ####
# 인덱스에 명목변수 적용

# 혈액형별 평균

tapply(DF$Height, DF$BloodType, mean)




# 성별별 평균

tapply(DF$Height, DF$Gender, mean)



# 학년별 평균

tapply(DF$Height, DF$Grade, mean)











# aggregate(종속변수 ~ 독립변수, 데이터, 적용함수) ####
# 종속변수 : 키
# 독립변수 : 혈액형

# 종속변수 ~ 독립변수

aggregate(Height ~ BloodType, DF, mean)



# 종속변수 ~ 독립변수1 + 독립변수2 + ...

aggregate(Height ~ BloodType + Gender, DF, mean)



# cbind(종속변수1, 종속변수2, ...) ~ 독립변수

aggregate(cbind(Height, Weight) ~ BloodType, DF, mean)



# 다수의 종속변수 ~ 다수의 독립변수

aggregate(cbind(Height, Weight) ~ BloodType + Gender, DF, mean)





# 동일한 결과 반환 ####
# 입력데이터형 -> 출력데이터형

# 데이터프레임 -> 데이터프레임

aggregate(Height ~ BloodType, DF, mean)




# 리스트 -> 리스트

lapply(split(DF$Height, DF$BloodType), mean)



# 리스트 -> 벡터

sapply(split(DF$Height, DF$BloodType), mean)



# 벡터 -> 벡터

tapply(DF$Height, DF$BloodType, mean)



# 벡터 -> 리스트

by(DF$Height, DF$BloodType, mean)









# dplyr 패키지 ####
# %>% - 여러 함수를 연결/체이닝(Chaining)
# filter() - 변수에 조건을 주어 필터링
# select() - 데이터셋에서 특정 컬럼만 선택
# arrange() - 데이터를 오름차순, 내림차순 정렬
# mutate() - 기존의 변수를 사용하여 새로운 변수 생성
# summarise(with group_by)
#   - 주어진 데이터를 집계(min, max, mean, count)


# dplyr 패키지 설치
# install.packages("dplyr")

library(dplyr)

DF1 <- read.csv("[readData]/PII.csv")




# Filter() 함수 ####
# 변수에 조건을 주어 필터링

filter(DF1, Height > 170)


filter(DF1, Height > 170 & Weight > 70)


filter(DF1, Grade %in% c("2", "4"))










# select() 함수 ####
# 데이터셋에서 특정 컬럼만 선택

select(DF1, Name, Age, BloodType) %>% head()


select(DF1, -Grade, -Picture, -Height) %>% tail()


select(DF1, Name:Grade) %>% tail(7) %>% head(5)



# starts_with(), ends_with, contains()

select(DF1, starts_with("G")) %>% head()


select(DF1, ends_with("t")) %>% head()


select(DF1, contains("g", ignore.case = FALSE)) %>% head()








# 여러 함수를 연결 %>% ####

DF1 %>% 
  select(Name, BloodType, Height, Weight) %>% 
  filter(Height > 175)





# arrange() 함수 ####
# 데이터를 오름차순 또는 내림차순 정렬

DF1 %>% 
  select(Name, BloodType, Height, Weight) %>% 
  filter(Height > 175) %>% 
  arrange(Weight)



DF1 %>% 
  select(Name, BloodType, Height, Weight) %>% 
  filter(Height > 175) %>% 
  arrange(desc(Weight))






# mutate() 함수 ####
# 기존의 변수를 사용하여 새로운 변수 생성

DF1 %>% 
  select(Name, Height, Weight) %>% 
  mutate("BMI" = round(Weight/(Height/100)^2))



DF1 %>% 
  select(Name, Height, Weight) %>% 
  mutate("BMI" = round(Weight/(Height/100)^2)) %>% 
  arrange(desc(BMI)) %>% head(7)







# _all, _at, _if ####

# mutate()
# 지정된 추가 변수 생성

DF1 %>% 
  select(Age, Height, Weight) %>% head() %>%
  mutate(sqrt(Age), sqrt(Height), sqrt(Weight))



# mutate_all()
# 기존 모든 변수에 함수 적용(덮어쓰기)

DF1 %>% 
  select(Age, Height, Weight) %>% head() %>%
  mutate_all(sqrt)



# mutate_at()
# 지정된 변수에 함수 적용(덮어쓰기)
# starts_with(), ends_with, contains()

DF1 %>% 
  select(Age, Height, Weight) %>% head() %>%
  mutate_at(vars(Age), sqrt)


DF1 %>% 
  select(Age, Height, Weight) %>% head() %>%
  mutate_at(vars(starts_with("A")), sqrt)


DF1 %>% 
  select(Age, Height, Weight) %>% head() %>%
  mutate_at(2:3, sqrt)



# mutate_if()
# 특정 조건을 만족하는 변수에 함수 적용(덮어쓰기)

DF1 %>% 
  select(Age, ends_with("t")) %>% head() %>%
  mutate_if(funs(mean(.) > 100), sqrt)


DF1 %>% 
  select(contains("g", ignore.case = FALSE)) %>% 
  head() %>%  mutate_if(funs(mean(.) > 100), sqrt)









# summarise(), group_by() ####
# 주어진 데이터를 집계(min, max, mean, count)

DF1 %>% 
  group_by(Gender) %>% 
  summarise(average = mean(Height, na.rm = TRUE))


DF1 %>% 
  group_by(Grade) %>% 
  summarise_at(vars(Height, Weight), funs(mean), na.rm = TRUE)


DF1 %>% 
  group_by(Grade) %>% 
  summarise_at(vars(Height, Weight), funs(mean, max))




DF1 %>% 
  group_by(Gender, Grade) %>% 
  summarise(average = mean(Height, na.rm = TRUE))


DF1 %>% 
  group_by(Gender, Grade) %>% 
  summarise_at(vars(Height, Weight), funs(mean, max))


DF1 %>% 
  group_by(Gender, Grade) %>% 
  summarise_at(vars(ends_with("t")), funs(mean, max))







# join with dplyr ####

library(dplyr)

TD_1 <- read.csv("[readData]/TD_1.csv")

TD_2 <- read.csv("[readData]/TD_2.csv")



# full_join() : 합집합
# 양쪽에 최소 한번 이상 등장하는 데이터 모두 포함

full_join(TD_1, TD_2, by = c("Vendor" = "Company"))

full_join(TD_2, TD_1, by = c("Company" = "Vendor"))




# inner_join() : 교집합
# 양쪽에 공통으로 등장하는 데이터만 포함

inner_join(TD_1, TD_2, by = c("Vendor" = "Company"))

inner_join(TD_2, TD_1, by = c("Company" = "Vendor"))




# left_join() : TD_1 또는 TD_2 기준
# 양쪽 공통 데이터와 왼쪽에 존재하는 데이터 포함

left_join(TD_1, TD_2, by = c("Vendor" = "Company"))

left_join(TD_2, TD_1, by = c("Company" = "Vendor"))





# right_join() : TD_2 또는 TD_1 기준
# 양쪽 공통 데이터와 오른쪽에 존재하는 데이터 포함

right_join(TD_1, TD_2, by = c("Vendor" = "Company"))

right_join(TD_2, TD_1, by = c("Company" = "Vendor"))



# anti_join() : 여집합
# 양쪽 공통 데이터를 제외하고 왼쪽 데이터만 출력

anti_join(TD_1, TD_2, by = c("Vendor" = "Company"))

anti_join(TD_2, TD_1, by = c("Company" = "Vendor"))




# semi_join()
# 양쪽 공통 데이터에서 왼쪽 데이터만 출력

semi_join(TD_1, TD_2, by = c("Vendor" = "Company"))

semi_join(TD_2, TD_1, by = c("Company" = "Vendor"))














# sqldf 패키지 설치 ####

# install.packages("sqldf")

library(sqldf)


# 실습용 Table 생성

TB <- read.csv("[readData]/PII.csv")

str(TB)

head(TB)






# "SELECT * FROM Table_Name" ####
# Column 단위 처리

sqldf("SELECT Height FROM TB")

sqldf("SELECT Gender, Height FROM TB")

sqldf("SELECT Name, Height FROM TB")

sqldf("SELECT Gender, Height, BloodType FROM TB")

sqldf("SELECT * FROM TB") %>% head()








# "WHERE" 조건구문 ####

sqldf("SELECT * FROM TB WHERE Height > 175")







# "AND" 연산

sqldf("SELECT * FROM TB WHERE Height > 175 AND Weight < 75")

sqldf("SELECT * FROM TB WHERE Height > 175 AND Grade = 3")





# "OR" 연산

sqldf("SELECT * FROM TB WHERE Height > 175 OR Grade = 3")

sqldf("SELECT * FROM TB WHERE Height > 175 OR Age = 24")





# 명목형(Character) 연산

sqldf("SELECT * FROM TB WHERE Name = '강백호'")

sqldf("SELECT * FROM TB WHERE BloodType = 'B'")

sqldf("SELECT * FROM TB WHERE Height > 175 AND BloodType = 'B'")













# "IN" 연산자 ####
# "OR" 연산자와 유사

sqldf("SELECT * FROM TB WHERE Grade IN ('2', '4')")

sqldf("SELECT * FROM TB WHERE BloodType IN ('A', 'B', 'O')")




# "LIKE" 연산자 ####
# 특정 문자 시작, 끝, 포함하는 값을 추출

sqldf("SELECT * FROM TB WHERE Weight LIKE '5%'")

sqldf("SELECT * FROM TB WHERE Weight LIKE '7%'")

sqldf("SELECT * FROM TB WHERE Weight LIKE '%9'")

sqldf("SELECT * FROM TB WHERE Weight LIKE '%2'")

sqldf("SELECT * FROM TB WHERE Name LIKE '김%'")

sqldf("SELECT * FROM TB WHERE Name LIKE '%정'")

sqldf("SELECT * FROM TB WHERE Name LIKE '%소%'")




# "GROUP BY" 연산자 ####
# 중복값을 제외하고 1개의 고유값만 출력

sqldf("SELECT Grade FROM TB GROUP BY Grade")






# "SUM( )"

sqldf("SELECT Grade, SUM(Age), SUM(Height), SUM(Weight) 
      FROM TB 
      GROUP BY Grade")





# "AVG( )"

sqldf("SELECT Grade, AVG(Age), AVG(Height), AVG(Weight) 
      FROM TB 
      GROUP BY Grade")





# "HAVING" 조건추가

sqldf("SELECT Grade, AVG(Age), AVG(Height), AVG(Weight) 
      FROM TB 
      GROUP BY Grade
      HAVING AVG(Height) > 170")




# "ORDER BY" 연산자
# 오름차순 : ASC , 내림차순 : DESC

sqldf("SELECT * FROM TB WHERE Height > 170")






# "Height" 기준

sqldf("SELECT * FROM TB WHERE Height > 170 ORDER BY Height ASC")

sqldf("SELECT * FROM TB WHERE Height > 170 ORDER BY Height DESC")





# "Weight" 기준

sqldf("SELECT * FROM TB WHERE Height > 170 ORDER BY Weight ASC")

sqldf("SELECT * FROM TB WHERE Height > 170 ORDER BY Weight DESC")





# "AS" 연산자 ####
# 출력 Column Name 변경

sqldf("SELECT SUM(Height) AS SUM_Height FROM TB")

sqldf("SELECT AVG(Height) AS AVG_Height, AVG(Weight) AS AVG_Weight FROM TB")











  


# reshape2 패키지 ####

# reshape2 패키지 설치
# install.packages("reshape2")

library(reshape2)




# 실습데이터 읽어오기

vlm <- read.csv("[readData]/volume.csv")







# melt() 함수 실행
# Wide Matrix -> Long Matrix

# 이름 지정

melt(data = vlm, id.vars = "FY",
     variable.name = "Quarter")



# 이름 미지정

melt(data = vlm, id.vars = "FY")



# 객체값 지정

vlm_melt <- melt(data = vlm, id.vars = "FY",
                 variable.name = "Quarter")




# dcast() 함수 실행
# Long Matrix -> Wide Matrix
# 세로 ~ 가로

dcast(data = vlm_melt, FY ~ Quarter)



# Total 추가

Total <- apply(vlm[2:5], 1, sum)

dcast(data = vlm_melt, FY + Total ~ Quarter)



# Total 추가 with mutate()

library(dplyr)

vlm %>% mutate("Total" = Q1 + Q2 + Q3 + Q4)















# stringr 패키지 ####
# install.packages("stringr")

library(stringr)

retxt <- c("Sports", "SPORTS", "sports", "sPORTS",
           "Baseball1", "sports", "baseball2")



# str_detect() : 패턴이 포함된 단어 확인(TRUE/FALSE)

str_detect(retxt, "s")

str_detect(retxt, "[1-9]$")

str_detect(retxt, "^[sS]")

str_detect(retxt, regex("^s", ignore_case = TRUE))

str_detect(retxt, "[Ba]")

str_detect(retxt, regex("b", ignore_case = TRUE))







# str_length() : 문자열의 길이 확인

str_length(retxt)


str_length("Big Data R Program")




# str_count() : 문자열별 패턴 포함 횟수 확인

str_count(retxt, "s")

str_count(retxt, "[sS]")

str_count(retxt, regex("s", ignore_case = TRUE))



# str_trim() : 문자열 앞/뒤 공백문자(" ") 제거

str_trim(" Big Data R Program ")

str_trim("Big Data R Program ")

str_trim(" Big Data R Program")



# str_c() : 문자열 합치기

str_c("Big ", "Data")

str_c("Big Data ", "R Program")

str_c(retxt, collapse = " ")

str_c(retxt, collapse = ":")

str_c("retxt : ", retxt)

str_c(retxt, " AND ", retxt)







# str_replace() : 주어진 패턴을 지정된 문자로 변경

str_replace(retxt, "a", "?")

str_replace(retxt, "[sS]", "*")





# str_split() : 주어진 패턴 문자로 데이터 분리

qoo <- "Big_Data_R_Program"

str_split(qoo, "_")





# str_sub() : 문자열 잘라내기

str_sub(retxt, start = 2, end = 4)

str_sub("replacement", start = 1, end = 7)

str_sub("replacement", start = -4, end = -1)

str_sub("replacement", start = 8)

str_sub("replacement", start = -4)









# 실습용 데이터 생성

ID <- c("001", "002", "003", "004", "005", "006", "007", "008", "009", "010") 
SCORE <- c(54, 100, 70, 88, 61, 77, 95, 0, 66, 75)

DF <- data.frame(ID, SCORE)
DF



# cut() ####
# break : 구간값 지정 벡터
# include.lowst : 최소값도 적용
# right = TRUE  : 60 < X <= 70
# right = FALSE : 60 <= X < 70
# labels  : 구간이름 지정 벡터
# dig.lab : labels 미지정시 구간값 출력
# ordered_result : labels(구간값)을 순서형으로 지정


# break & labels
# 8번째 값 NA

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    labels = c("F", "D", "C", "B", "A"))




# include.lowest
# 8번째 값 처리

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    labels = c("F", "D", "C", "B", "A"),
    include.lowest = TRUE)



# labels 미지정
# 구간값으로 처리

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    include.lowest = TRUE)




# right = TRUE  : 60 < X <= 70

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    labels = c("F", "D", "C", "B", "A"),
    include.lowest = TRUE,
    right = TRUE)



# right = FALSE : 60 <= X < 70

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    labels = c("F", "D", "C", "B", "A"),
    include.lowest = TRUE,
    right = FALSE)



# ordered_result : labels(구간값)을 순서형으로 지정

cut(SCORE, 
    breaks = c(0, 60, 70, 80, 90, 100),
    labels = c("F", "D", "C", "B", "A"),
    include.lowest = TRUE,
    right = TRUE,
    ordered_result = TRUE)


# DF에 cut() 실행결과 추가

DF$cut_result <- cut(SCORE, 
                     breaks = c(0, 60, 70, 80, 90, 100),
                     labels = c("F", "D", "C", "B", "A"),
                     include.lowest = TRUE,
                     right = TRUE)



DF









# ifelse() ####

ifelse(SCORE < 60, "F",
       ifelse(SCORE >= 60 & SCORE < 70, "D",
              ifelse(SCORE >= 70 & SCORE < 80, "C",
                     ifelse(SCORE >= 80 & SCORE < 90, "B", "A"))))




# DF에 ifelse() 실행결과 추가

DF$ifelse_result <- ifelse(SCORE < 60, "F",
                           ifelse(SCORE >= 60 & SCORE < 70, "D",
                                  ifelse(SCORE >= 70 & SCORE < 80, "C",
                                         ifelse(SCORE >= 80 & SCORE < 90, "B", "A"))))


DF











# split() ####
# DataFrame에 적용

split(DF, DF$cut_result)


split(DF, DF$ifelse_result)



# Vector에 적용

split(DF$SCORE, DF$cut_result)


split(DF$SCORE, DF$ifelse_result)




# *apply() 적용
# lapply()

lapply(split(DF$SCORE, DF$cut_result), length)
lapply(split(DF$SCORE, DF$cut_result), mean)

lapply(split(DF$SCORE, DF$ifelse_result), length)
lapply(split(DF$SCORE, DF$ifelse_result), mean)


# tapply()

tapply(DF$SCORE, DF$cut_result, length)
tapply(DF$SCORE, DF$cut_result, mean)

tapply(DF$SCORE, DF$ifelse_result, length)
tapply(DF$SCORE, DF$ifelse_result, mean)



# unsplit()

unsplit(split(DF, DF$cut_result), DF$cut_result)

















# unique() ####

# Vector에 적용

DF$ifelse_result
unique(DF$cut_result)
unique(DF$ifelse_result)



# DataFrame에 적용

DF

duplicated(DF$cut_result)
DF[!duplicated(DF$cut_result), ]


duplicated(DF$ifelse_result)
DF[!duplicated(DF$ifelse_result), ]










# The End ####