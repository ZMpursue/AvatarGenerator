import requests
from bs4 import BeautifulSoup
import os
import traceback
import threading

#imglist = os.listdir('./imgs')
def download(url, filename):
    if os.path.isfile(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=10000)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        #imglist.append(filename)
        return filename
    except KeyboardInterrupt:
        print(KeyboardInterrupt)
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        print(Exception)
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)


if os.path.exists('imgs2') is False:
    os.makedirs('imgs2')

class thread(threading.Thread):
    def __init__(self, threadID,flag, begin, end):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.flag = flag
        self.begin = begin
        self.end = end
    def run(self):
        if self.flag:
            url = 'https://safebooru.donmai.us/posts?page='
            classname = "has-cropped-true"
        else:
            url = 'http://konachan.net/post?page=%d&tags='
            classname = "preview"
        for i in range(self.begin, self.end + 1):
            url = url + str(i)
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'html.parser')
            for img in soup.find_all('img', class_=classname):
                target_url = img['src']
                filename = os.path.join('imgs2', target_url.split('/')[-1])
                download(target_url, filename)
            print('Thead%d : %d / %d' % (self.threadID,i, self.end))

if __name__ == '__main__':
    T1 = thread(1, True, 1, 8000)
    #T2 = thread(2, True, 3001, 6000)
    #T3 = thread(3, True, 6001, 9000)
    T4 = thread(4, False, 1, 8000)
    #T5 = thread(5, False, 3001, 6000)
    #T6 = thread(6, False, 6001, 9000)
    T1.start()
    #T2.start()
    #T3.start()
    T4.start()
    #T5.start()
    #T6.start()
    T1.join()
    #T2.join()
    #T3.join()
    T4.join()
    #T5.join()
    #T6.join()
