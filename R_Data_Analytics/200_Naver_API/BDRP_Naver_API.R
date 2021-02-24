# install.packages("RCurl")
# install.packages("XML")

library(RCurl)
library(XML)





# 블로그 검색 API ####

searchUrl <- "https://openapi.naver.com/v1/search/blog.xml"



# 클라이언트 ID와 시크릿
# 본인의 ID/Secret 입력

Client_ID <- "본인 ID 입력"
Client_Secret <- "본인 Secret 입력"



# 검색어 

query <- URLencode(iconv("코로나", "euc-kr", "UTF-8"))



# 블로그 검색
# &display 최대값 100

url <- paste(searchUrl, "?query=", query, "&display=100", sep="")

doc <- getURL(url, 
              httpheader = c("Content-Type" = "application/xml",
                             "X-Naver-Client-Id" = Client_ID,
                             "X-Naver-Client-Secret" = Client_Secret))



# 블로그 내용에 대한 리스트 만들기		

doc2 <- htmlParse(doc, encoding = "UTF-8")

text <- xpathSApply(doc2, "//item/description", xmlValue) 

text











# 뉴스 검색 API ####

searchNrl <- "https://openapi.naver.com/v1/search/news.xml"



# 검색어 

query <- URLencode(iconv("코로나", "euc-kr", "UTF-8"))



# 뉴스 검색
# &display 최대값 100

nrl <- paste(searchNrl, "?query=", query, "&display=100", sep="")

ndoc <- getURL(nrl, 
              httpheader = c("Content-Type" = "application/xml",
                             "X-Naver-Client-Id" = Client_ID,
                             "X-Naver-Client-Secret" = Client_Secret))



# 뉴스 내용에 대한 리스트 만들기		

ndoc2 <- htmlParse(ndoc, encoding = "UTF-8")

ntext <- xpathSApply(ndoc2, "//item/description", xmlValue) 

ntext









# 카페 검색 API ####

searchCrl <- "https://openapi.naver.com/v1/search/cafearticle.xml"



# 검색어 

query <- URLencode(iconv("코로나", "euc-kr", "UTF-8"))



# 카페글 검색
# &display 최대값 100

crl <- paste(searchCrl, "?query=", query, "&display=100", sep="")

cdoc <- getURL(crl, 
               httpheader = c("Content-Type" = "application/xml",
                              "X-Naver-Client-Id" = Client_ID,
                              "X-Naver-Client-Secret" = Client_Secret))



# 카페글 내용에 대한 리스트 만들기		

cdoc2 <- htmlParse(cdoc, encoding = "UTF-8")

ctext <- xpathSApply(cdoc2, "//item/description", xmlValue) 

ctext









# The End ####









# SSH Issue ####

library(stringr)
library(httr)



# 검색 api 설정

api_url <-  "https://openapi.naver.com/v1/search/webkr.xml"



# 검색어 지정

query <- URLencode(iconv("코로나", to = "UTF-8"))



# 검색 실행

query <- str_c("?query=", query)

result  <- GET(str_c(api_url, query,  "&display=100"), 
             add_headers("X-Naver-Client-Id" = Client_ID,
                         "X-Naver-Client-Secret" = Client_Secret))



# 검색결과 확인

ndoc <- htmlParse(result, encoding="UTF-8")

ntext <- xpathSApply(ndoc, "//item/description", xmlValue) 

ntext




# The End ####