# ggplot2 package #### 

# install.packages("ggplot2")
# install.packages("ggthemes")

library(ggplot2)
library(ggthemes)



# 그래프 출력함수 사용 ####

ggplot(iris, 
       aes(x = Sepal.Length, y = Sepal.Width, colour = Species)) + 
       geom_point() + theme_wsj()





# 객체지정하여 사용하기 ####
# g1 객체지정

g1 <- ggplot(iris, 
             aes(x = Sepal.Length, y = Sepal.Width, colour = Species))



# g2 객체지정

g2 <- geom_point()



# g3 객체지정

g3 <- theme_bw()



# g1 + g2 + g3 그래프 출력

g1 + g2 + g3



# 테마변경

g1 + g2 + theme_fivethirtyeight()











# geom_point() and geom_line() ####

BD <- read.csv("[readData]/BB_plyr_2.csv")



# 해더정보 확인

colnames(BD)



# 데이터 구조 확인

str(BD)

head(BD)



# 그래픽 객체지정

g1 <- ggplot(BD, aes(x = 연도, y = 안타))



# 포인트 출력

g1 + geom_point()



# 라인 출력

g1 + geom_line()



# 포인트 + 라인 출력

g1 + geom_point() + geom_line() 



# 사이즈 및 색 지정 ####

g1 + 
  geom_point(size = 5) + geom_line(size = 1)



g1 + 
  geom_point(size = 5, colour = "seagreen") + 
  geom_line(size = 1, colour = "gray55")






# aes( ) 색 지정 및 점모양 변경 ####

BD2 <- read.csv("[readData]/KBO2.csv")

str(BD2)

(BD2$팀명)



ggplot(BD2, aes(x = 순위, y = 안타, colour = 팀명)) +
  geom_point(size = 2)



ggplot(BD2, aes(x = 순위, y = 안타, colour = 팀명)) +
  geom_point(size = 3, shape = 11)



ggplot(BD2, aes(x = 순위, y = 안타, colour = 팀명)) +
  geom_point(size = 5, shape = "$")








# plotly Package ####
# install.packages("plotly")

library(plotly)

g2 <- ggplot(BD2, aes(x = 순위, y = 안타, colour = 팀명)) +
          geom_point(size = 2)



# ggplotly()
# 동적그래프 생성

ggplotly(g2)






# geom_area() ####

# 라인과 영역으로 그래프 출력

ggplot(BD, aes(x = 연도, y = 이루타)) + geom_line()

ggplot(BD, aes(x = 연도, y = 이루타)) + geom_area()



# 색 채우기

ggplot(BD, aes(x = 연도, y = 이루타)) + 
  geom_area(fill = "tan")



# 색 채우기와 경계선 긋기

ggplot(BD, aes(x = 연도, y = 이루타)) + 
  geom_area(fill = "tomato") + 
  geom_line(size = 2, colour = "gray33")



# alpha : 투명도 지정

ggplot(BD, aes(x = 연도, y = 이루타)) + 
  geom_area(fill = "tomato", alpha = 0.5) + 
  geom_line(size = 2, colour = "gray33")










# geom_histogram() ####
# 객체지정

g1 <- ggplot(iris, aes(x = Sepal.Length))



# 히스토그램 출력

g1 + 
  geom_histogram(binwidth = 0.2, 
                 fill = "peru")



# 색 채우기 적용

g1 + 
  geom_histogram(aes(fill = Species), 
                 binwidth = 0.2)










# geom_text() ####

DF <- read.csv("[readData]/PII.csv")

str(DF)

g1 <- ggplot(DF, aes(x = Height, y = Weight))


# 이름 표시하기

g1 + 
  geom_point(aes(colour = BloodType), size = 5) + 
  geom_text(aes(label = Name))



g1 + 
  geom_point(aes(colour = BloodType), size = 5) + 
  geom_text(aes(label = Name), 
            vjust = -1.1,
            colour = "gray33")








# geom_bar() ####

ggplot(DF, aes(x = BloodType)) + geom_bar()




# fill로 색지정

ggplot(DF, aes(x = BloodType, fill = Gender)) + 
  geom_bar()



# 성별 막대그래프 분리

ggplot(DF, aes(x = BloodType, fill = Gender)) + 
  geom_bar(position = "dodge")



# 막대그래프 폭 지정

ggplot(DF, aes(x = BloodType, fill = Gender)) + 
  geom_bar(position = "dodge", width = 0.3, stat = "count")



# stat = "identity"

ggplot(DF, aes(x = Name, y = Height, fill = BloodType)) +
  geom_bar(stat = "identity")

ggplot(DF, aes(x = Name, y = Height, fill = BloodType)) +
  geom_bar(stat = "identity", width = 0.3)



# geom_text() & theme()

ggplot(DF, aes(x = Name, y = Height, fill = BloodType)) +
  geom_bar(stat = "identity", width = 0.3) +
  geom_text(aes(label = paste(Grade, "학년")),
            vjust = -1, color = "gray35", size = 3) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
  














# geom_boxplot() ####

g1 <- ggplot(DF, aes(x = Gender, y = Height))

g1 + geom_boxplot()

g1 + geom_boxplot(aes(fill = Gender))

g1 + geom_boxplot(fill = c("skyblue", "pink"))



g2 <- ggplot(DF, aes(x = BloodType, y = Height))

g2 + geom_boxplot(aes(fill = BloodType))

g2 + geom_boxplot(aes(fill = BloodType),
                  outlier.colour = "red",
                  outlier.shape = "*",
                  outlier.size = 30)










# facet_grid(종속변수~독립변수)  ####
# 명목형 변수를 기준으로 별도의 그래프 작성

g1 <- ggplot(DF, aes(x = Height, y = Weight, 
                     colour = BloodType))



# x축 기준

g1 + 
  geom_line(size = 1) + 
  geom_point(size = 5) + 
  facet_grid(. ~ Gender)




# y축 기준

g1 + 
  geom_line(size = 1) + 
  geom_point(size = 5) + 
  facet_grid(Gender ~ .)




# scales = "free"를 적용하여 
# 명목변수 별 각각의 범위를 허용
# x축기준(facet_wrap( ))

g1 + 
  geom_line(size = 1) + 
  geom_point(size = 5) + 
  facet_wrap(. ~ Gender, scales = "free")



# y축기준(facet_grid( ))

g1 + 
  geom_line(size = 1) + 
  geom_point(size = 5) + 
  facet_grid(Gender ~ ., scales = "free")





# gridExtra::grid.arrange() ####

# gridExtra 패키지 설치

# install.packages("gridExtra")



# 데이터 읽어오기

DF <- read.csv("[readData]/PII.csv")



# 시각화용 데이터 지정

gg <- ggplot(DF, aes(x = Height, y = Weight))



# 산점도

g1 <- gg + geom_point(size = 4, colour = "tomato")




# 선 그래프

g2 <- gg + geom_line(size = 2, colour = "peru")




# 상자 그래프

g3 <- gg + geom_boxplot(size = 2, colour = "seagreen")




# 회귀선

g4 <- gg + geom_smooth(size = 2, colour = "blue")






# 최종 그래프 출력

gridExtra::grid.arrange(g1, g2,
                        g3, g4,
                        nrow = 2)





# The End ####