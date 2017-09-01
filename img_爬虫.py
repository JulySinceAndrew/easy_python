from urllib import request
import urllib
from html.parser import HTMLParser
import re
from html.entities import name2codepoint

class MyHTMLhander(HTMLParser):
    def handle_startendtag(self, tag, attrs):
        if tag=='img':
            print(attrs)
        pass

def get_Img(html):
    #condition=r'class="BDE_Image" src=\"(.+?\.[a-z]+)\"'
    condition=r'<img.*?src=\"(http.+?\..+?)\"'
    re_img=re.compile(condition)
    img_list=re.findall(re_img,html)
    i=0
    for x in img_list:
        print(x)
        try:
            request.urlretrieve(x,'img/%d.jpg'%i)
        except urllib.error.HTTPError:
            i=i-1
        i=i+1
    print(i)

def get_html(url):
    req=request.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11')
    with request.urlopen(req) as f:
        html=f.read()
        html=html.decode('utf-8')
        print(html)
        return html

s='https://www.douban.com/'
get_Img(get_html(s))