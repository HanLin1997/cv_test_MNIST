import os
import cv2
import torchvision.datasets.mnist as mnist

root = "/Users/linhan/Desktop/cv/mnist/data/MNIST/raw"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )


train_path="/Users/linhan/Desktop/cv/mnist/dataset/train/"
if(not os.path.exists(train_path)):
    os.makedirs(train_path)
for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
    img_path=train_path+str(i)+'.jpg'
    cv2.imwrite(img_path,img.numpy())

test_path = "/Users/linhan/Desktop/cv/mnist/dataset/test/"
if (not os.path.exists(test_path)):
    os.makedirs(test_path)
for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
    img_path = test_path+ str(i) + '.jpg'
    cv2.imwrite(img_path, img.numpy())
