import cv2
import numpy as np
from keras.models import model_from_json

model_path = './model/'
img_size = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)

# 从json中加载模型
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# 加载模型权重
model.load_weights(model_path + 'model_weight.h5')

if __name__ == '__main__':
    # 创建VideoCapture对象
    capture = cv2.VideoCapture(0)

    # 使用opencv的人脸分类器
    cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')

    while True:
        ret, frame = capture.read()

        # 灰度化处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 识别人脸位置
        faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=1, minSize=(120, 120))

        if len(faceLands) > 0:
            for faceLand in faceLands:
                x, y, w, h = faceLand
                images = []
                result = np.array([0.0] * num_class)

                # 裁剪出脸部图像
                image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
                image = image / 255.0
                images.append(image)
                images.append(cv2.flip(image, 1))
                images.append(cv2.resize(image[2:45, :], (img_size, img_size)))

                for image in images:
                    image = image.reshape(1, img_size, img_size, 1)
                    predict_lists = model.predict_proba(image, batch_size=32, verbose=1)
                    # print(predict_lists)
                    result += np.array([predict for predict_list in predict_lists
                                        for predict in predict_list])
                    # print(result)
                    emotion = emotion_labels[int(np.argmax(result))]
                    print("Emotion:", emotion)
                    cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20),
                                  (0, 0, 255), thickness=2)
                    cv2.putText(frame, '%s' % emotion, (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

            cv2.imshow('EmotionClassify', frame)
            if cv2.waitKey(60) == ord('q'):
                break

    # 释放摄像头并销毁所有窗口
    capture.release()
    cv2.destroyAllWindows()
