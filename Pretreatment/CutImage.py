import cv2
import sys
import os.path
import threading

cascade_file="./lbpcascade_animeface.xml"

def detect(filename,outdir):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(96,96))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (96, 96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("faces/" + save_filename, face)

class thread(threading.Thread):
    def __init__(self, threadID, path, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.files = files
    def run(self):
        for file in self.files:
            detect(self.path + file,self.path)

def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o

if __name__ == '__main__':
    path = '../data/train/'
    files =  os.listdir(path)
    files = cutArray(files,6)
    T1 = thread(1, path, files[0])
    T2 = thread(2, path, files[1])
    T3 = thread(3, path, files[2])
    T4 = thread(4, path, files[3])
    T5 = thread(5, path, files[4])
    T6 = thread(6, path, files[5])
    T1.start()
    T2.start()
    T3.start()
    T4.start()
    T5.start()
    T6.start()
    T1.join()
    T2.join()
    T3.join()
    T4.join()
    T5.join()
    T6.join()

