# 필요한 패키지 설치 ####
# install.packages(c("rvest", "httr", "dplyr"))

library(rvest)
library(httr)
library(dplyr)




# HTTP 요청 ####
# NAVER 메인페이지

NAVER <- GET('https://datalab.naver.com/keyword/realtimeList.naver?where=main')






# HTTP 응답코드 ####

status_code(NAVER)







# NAVER 객체 내용 확인 ####

content(NAVER, 
        as = 'text', 
        encoding = 'UTF-8') %>% cat()





# html_document 생성 ####

html <- read_html(NAVER)


html




# CSS 요소 선택 ####
# <span class="item_title">검색어</span>

span <- html_nodes(html,
                   css = 'span.item_title')

span



# TOP_20 추출 ####

TOP_20 <- html_text(span)


TOP_20




# The End ####