# 작업디렉토리 확인 #### 

getwd()







# 작업디렉토리 지정 ####

setwd("C:/Users")

getwd()

dir()



# 작업디렉토리 재지정 

setwd("~")

getwd()

dir()













# readLines() ####

readLines("[readData]/TestFile_1.txt")

readLines("[readData]/TestFile_2.txt")





# read.table() ####

# 해더 자동 생성

read.table("[readData]/FileIO.txt")



# 해더 자동 생성하지 않음

read.table("[readData]/FileIO.txt", 
            header = TRUE)








# read.csv() ####

# 해더 자동 인식

read.csv("[readData]/FileIO_1.csv")



# 해더 없는 경우
# Error

read.csv("[readData]/FileIO_2.csv")



# 해더 추가 생성

read.csv("[readData]/FileIO_2.csv", 
          header = FALSE)











# write.csv() ####

DF <- read.csv("[readData]/FileIO_1.csv")

CSV <- DF[-5, 1:3]

write.csv(CSV, 
          file = "[writeData]/001_outCSV.csv",
          row.names = FALSE)

read.csv("[writeData]/001_outCSV.csv")









# read.xlsx() ####

# install.packages("xlsx")

library(xlsx)



read.xlsx(file = "[readData]/PII.xlsx",
          sheetName = "Sheet1",
          encoding = "UTF-8")




# write.xlsx() ####

# Sheet 이름 지정 안함

write.xlsx(CSV, 
           file = "[writeData]/002_outXLSX.xlsx", 
           row.names = FALSE)



read.xlsx(file = "[writeData]/002_outXLSX.xlsx", 
          sheetName = "Sheet1")



# Sheet 이름 지정함

write.xlsx(DF,
           file = "[writeData]/002_outXLSX.xlsx",
           sheetName = "DF_list",
           col.names = TRUE,
           row.names = FALSE,
           append = TRUE)


read.xlsx(file = "[writeData]/002_outXLSX.xlsx", 
          sheetName = "DF_list")








# The End ####