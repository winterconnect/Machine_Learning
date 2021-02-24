# 조건문(흐름제어) #### 

n <- 3



# Type1 ####

if(n == 3){
  print("n은 3입니다.")
} else {
  print("n은 3이 아닙니다.")
}



# Type2 ####

ifelse(n == 3, "n은 3.", "n은 5.")











n <- 5

# Type1

if(n == 3){
  print("n은 3입니다.")
} else {
  print("n은 3이 아닙니다.")
}



# Type2

ifelse(n == 3, "n은 3.", "n은 5.")















# 반복문(반복실행) ####
# for() ####

for(num in 1:10){ 
  print(num) 
}


for(num in c(1, 5, 9)){
  print(num + 10)
}


for(num in 3:7){
  print("Hello World")
}




# while() ####

koo <- 1

while(koo < 10){
  print(koo)
  koo <- koo + 1  
}












# repeat() ####

koo <- 1

repeat{
  print(koo)
  if(koo == 9) { break }
  koo <- koo + 1
}











# switch() ####

opt <- "f"

switch(opt,
       a = print("opt is a"),
       b = print("opt is b"),
       c = print("opt is c"),
       d = print("opt is d"),
       print("opt is not a,b,c,d"))










# 반복문 + 조건문 ####

for(n in 1:4){
  if(n == 3){
    cat("입력된 값", n, "은 3 입니다.\n")
  } else {
    cat("입력된 값", n, "은 3 이 아닙니다.\n")
  }
}











# 사용자 지정함수 ####
# 함수[특정 기능 반복 사용]


# 인자가 없는 함수선언

myfunc1 <- function() {
  x <- 11
  y <- 22
  return(x + y)
}


myfunc1()






# 인자가 있는 함수선언

myfunc2 <- function(x, y) {
  XX <- x
  YY <- y
  return(sum(XX, YY))
}


myfunc2(11, 22)










# 함수에서 함수 호출

myfunc3 <- function(x, y) {
  x3 <- x+1
  y3 <- y+1
  
  x4 <- myfunc2(x3, y3)
  return(x4)
}


myfunc3(10, 21)








# 함수 외부객체 호출하여 사용

x <- 9
x
ls()

myfunc5 <- function() {
  x <<- 11
  y <- 22
  return(x + y)
}

x
myfunc5()
x





# 파일 저장 함수 ####
# 특정기능 함수 파일로 저장 후 재사용
# 다른 스크립트 실행 중에 읽어 들여서 사용

# 새로운 함수 선언

newfunc1 <- function(x) {
  return(x * x)
}

newfunc1(3)



# 작업디렉토리에 함수를 파일로 저장

getwd()

save(newfunc1, file = "myfunc.Rdata")



# 메모리에서 함수 삭제(에러발생)

rm("newfunc1")

newfunc1(3)



# 파일로부터 지정된 함수 읽어들이기

load("myfunc.Rdata")

newfunc1(3)





# 여러개의 함수 파일에 저장하기

newfunc2 <- function(x) {
  return(x + x)
  }


newfunc2(3)


save(newfunc1, newfunc2, file = "myfunc.Rdata")









# 최종 Test

rm("newfunc1")
rm("newfunc2")

newfunc1(3)
newfunc2(3)



load("myfunc.Rdata")

newfunc1(3)
newfunc2(3)






# The End ####