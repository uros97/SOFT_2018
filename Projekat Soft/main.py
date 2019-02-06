import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from number import Number
from all_numbers_video import  All_numbers
import linija as linija
# keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import test as testiranje


#rad sa slikom
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_bin(image_gs):
    #height, width = image_gs.shape[0:2]
    #image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255-image


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3,3), np.uint8) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def copy_number(img, number):
    '''Kopira prepoznat broj na skaliranu povrsinu, da bi se slagao sa brojevima kojima je klasifikator obucavan'''
    region = np.zeros((28, 28)).astype('float32')
    granice = number.granice
    x = granice[0] - 3
    y = granice[1] - 3
    w = granice[2] + 3
    h = granice[3] + 3
    modified_h = 28.0 - h
    modified_w = 28.0 - w
    y_off = int(modified_h / 2.0)
    x_off = int(modified_w / 2.0)
    for j in range(0, w):
        for k in range(0, h):
            flag = 0 <= y + k < img.shape[0] and 0 <= x + j < img.shape[1]
            if flag:
                region_x = y_off + k
                region_y = x_off + j
                img_x = y + k
                img_y = x + j
                region[region_x, region_y] = img[img_x, img_y] / 255.0
    return region


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    granice = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        if h > 8:
            trenutne = x, y, w, h
            granice.append(Number((x, y, w, h), False))
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    return image_orig, granice


def analize_video(video_path, klas, all_copies):
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    ok, frame_1 = cap.read()
    cap.set(1, frame_num)  # indeksiranje frejmova
    all_frames = []
    rezultat = 0
    svi_brojevi = All_numbers()
    moja_linija = linija.pronadji_liniju(frame_1)
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        # plt.imshow(frame)
        # ako frejm nije zahvacen
        if not ret_val:
            break
        all_frames.append(frame)
        maska = linija.get_mask(frame, 127, 127, 127, 255, 255, 255)
        bez_suma = cv2.bitwise_and(frame, frame, mask=maska)
        bez_suma = image_bin(image_gray(bez_suma))
        selected_frame,brojevi = select_roi(frame_1, bez_suma)
        brojevi = svi_brojevi.update(brojevi)
        for granica in brojevi:
            flag = not granica.prosao_liniju and linija.prosao_broj(moja_linija, granica)
            if flag:
                copy = copy_number(bez_suma, granica)
                all_copies.append(copy)
                ulaz = copy.reshape(1, 784).astype('float32')
                izlaz = int(klas.findNearest(ulaz, k=1)[0])
                #ulaz = copy.reshape(1, 28, 28, 1).astype('float32')
                #izlaz = klas.predict(ulaz)
                rezultat += izlaz
                granica.prosao_liniju = True
                print('Izlaz', izlaz)
    cap.release()
    return all_frames, rezultat


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def train_knn():

    train, test = load_mnist()

    x_train = train[0].reshape(60000, 784)
    y_train = train[1].astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255

    knn = cv2.ml.KNearest_create()
    print("Train KNN")
    knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    print("Done...")
    return knn


def train_neural_network():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=3)

    return model


#klasifikator = train_neural_network()
klasifikator = train_knn()
all_copies = []

with open('out.txt', 'w') as file:
    file.write('RA 176/2015 Uros Jakovljevic\n')
    file.write('Video\tsuma\t\n')

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    all_frames, rezultat = analize_video('data/videos/video-' + str(i) + '.avi', klasifikator, all_copies)
    with open('out.txt', 'a') as file:
        file.write('video-' + str(i) + '\t' + str(rezultat) + '\n')
    print(rezultat)

testiranje.test()
