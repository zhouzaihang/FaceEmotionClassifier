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

# 加载emotion
emotion_images = {}
for emoji in emotion_labels:
    emotion_images[emoji] = cv2.imread("./emoji/" + emoji + ".png", -1)


def face2emoji(face, emotion_index, position):
    x, y, w, h = position
    emotion_image = cv2.resize(emotion_images[emotion_index], (w, h))
    overlay_img = emotion_image[:, :, :3]/255.0
    overlay_bg = emotion_image[:, :, 3:]/255.0
    background = (1.0 - overlay_bg)
    face_part = (face[y:y + h, x:x + w]/255.0) * background
    overlay_part = overlay_img * overlay_bg

    face[y:y + h, x:x + w] = cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)

    return face


# 创建VideoCapture对象
capture = cv2.VideoCapture(0)

# 使用opencv的人脸分类器
cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')

while True:
    ret, frame = capture.read()

    # 灰度化处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 呈现用emoji替代后的画面
    emoji_show = frame.copy()

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
            image = image.reshape(1, img_size, img_size, 1)

            # 调用模型预测情绪
            predict_lists = model.predict_proba(image, batch_size=32, verbose=1)
            # print(predict_lists)
            result += np.array([predict for predict_list in predict_lists
                                for predict in predict_list])
            # print(result)
            emotion = emotion_labels[int(np.argmax(result))]
            print("Emotion:", emotion)

            emoji = face2emoji(emoji_show, emotion, (x, y, w, h))
            cv2.imshow("Emotion", emoji)

            # 框出脸部并且写上标签
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20),
                          (0, 255, 255), thickness=10)
            cv2.putText(frame, '%s' % emotion, (x, y - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, 30)
            cv2.imshow('Face', frame)

        if cv2.waitKey(60) == ord('q'):
            break

# 释放摄像头并销毁所有窗口
capture.release()
cv2.destroyAllWindows()
