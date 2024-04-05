#.................................face_recognition(person)............................................#
import os
import time 
import cv2
import face_recognition
import numpy as np
from skimage import io
from sklearn import svm
from joblib import dump, load
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report,accuracy_score


class myclass:

    path = 'E:/billboard/Dataset/'

    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(folder)
    print(folders)
    known_face_encodings=[]
    known_face_names=[]
    for f in folders:
        ifile=f'Dataset/{f}/{f}_0.jpg'
        print(ifile)
        ru_image = face_recognition.load_image_file(ifile)
        ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
        known_face_encodings.append(ru_face_encoding)
        known_face_names.append(f)
    print('done extraction')

    clf = svm.SVC(gamma='scale')
    clf.fit(known_face_encodings,known_face_names)
    dump(clf,'SVM.Model') # stores JSON data directly to file
    print('done training') 

    TestData="Test"
    while True:
        
        for(direcpath,direcnames,files) in os.walk(TestData):
            f=open('new_user.txt')
            data = f.read()
            f.close()
            if data != '':
                print(data)
                f=open('new_user.txt','w')
                f.write('')
                f.close() 
        
                ru_image = face_recognition.load_image_file(f'Dataset/{data}/{data}.jpg')
                ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
                known_face_encodings.append(ru_face_encoding)
                known_face_names.append(data)
            f=open('readdata.txt')
            data = f.read()
            f.close()
            if data == 'read':
                print(data)
                f=open('readdata.txt','w')
                f.write('busy')
                f.close()
                time.sleep(2)
                exists = os.path.isfile(TestData+'//a.jpg')
                if exists:
                    try:
                        frame = (TestData+'//a.jpg')

                        unknown_image = face_recognition.load_image_file(frame)
                        face_locations = face_recognition.face_locations(unknown_image)
                        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

                        pil_image = Image.fromarray(unknown_image)
                        draw = ImageDraw.Draw(pil_image)
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            print('inside for')
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown Person"

                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                            text_width, text_height = draw.textsize(name)
                            print("The Person is:",name)
                            f = open("output.txt","w")
                            print((name),file = f)
                            f.close()
                            if name == "Unknown Person":
                                path = (TestData+'//a.jpg')
                                un_image = cv2.imread(path)
                                cv2.imwrite("Unknown.jpg",un_image)
                                #pil_image.save("Unknown.jpg")
                                f=open("result.txt","w")
                                print(("unknown"),file = f)
                                f.close()
                                
                                path = 'E:/billboard/Dataset/'
                                
                                folders = []

                                # r=root, d=directories, f = files
                                for r, d, f in os.walk(path):
                                    for folder in d:
                                        folders.append(folder)
                                print(folders)
                                os.mkdir(path+str(len(folders))) 
                                cv2.imwrite(path+str(len(folders))+"/"+str(len(folders))+"_0.jpg",un_image)
                                known_face_encodings=[]
                                known_face_names=[]
                                for f in folders:
                                    ifile=f'Dataset/{f}/{f}_0.jpg'
                                    ru_image = face_recognition.load_image_file(ifile)
                                    ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
                                    known_face_encodings.append(ru_face_encoding)
                                    known_face_names.append(f)
                                    clf = svm.SVC(gamma='scale')
                                    clf.fit(known_face_encodings,known_face_names)
                                    dump(clf,'SVM.Model')
                            else:
                                f=open("result.txt","w")
                                str1=f'face_matched:{name}'
                                print(str1,file = f)
                                f.close()
                            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

                        del draw
                        #pil_image.show()
                        pil_image.save("recognition/output.jpg")
                        os.remove(TestData+'//a.jpg')
                        time.sleep(5)
                        f=open('readdata.txt','w')
                        f.write('') 
                        f.close()
                    except:
                        print('could not read')
                        f=open('readdata.txt','w')
                        f.write('')
                        f.close()   


