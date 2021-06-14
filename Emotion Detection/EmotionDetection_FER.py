folder=r'C:\Users\12485\Desktop\NLP Project\Image Dataset\train\Test'
angry_count=0
total_angry_count=0
for filename in os.listdir(folder):
    #print(filename)
    img = cv2.imread(os.path.join(folder,filename))
    detector = FER(mtcnn=True)
    print(detector.detect_emotions(img))

    try:
        emotion,score=detector.top_emotion(img)
        print(emotion)
        if(emotion=='angry'):
            angry_count+=1
        total_angry_count+=1
    except:
        total_angry_count+=1
print(angry_count/total_angry_count)