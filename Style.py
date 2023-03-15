#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install --upgrade tensorflow_hub')


# In[103]:


get_ipython().system('pip install matplotlib')


# In[23]:


import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2


#使用 IPython.display.display 函数来进行图像的显示。
#from IPython.display import display


# In[24]:


model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


# In[25]:


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


# In[26]:


# 遍历指定文件夹中的所有图像文件并进行风格转换
content_images = []
for filename in os.listdir('Images1'):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 读取图像文件并进行预处理
        content_image = load_image(os.path.join('Images1', filename))
        content_images.append(content_image)
        
        # 显示处理后的图像
        plt.imshow(np.squeeze(content_image))
        plt.show()

style_image = load_image('image/output3.jpg')
plt.imshow(np.squeeze(style_image))
plt.show()


# In[27]:


# 创建一个空列表，用于存储stylized_images
stylized_images = []

# 遍历每个content_image，并将其与style_image一起传递到模型中
for content_image in content_images:
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_images.append(stylized_image)


# In[28]:


#stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]


# In[52]:


a = []
a.append((r"I w**? *o"))
a.append((r"go **t"))
a.append((r"##ou#"))
a.append((r"#ant t*"))
#plt.axes([0.025,0.025,0.9,0.9])

#for i in range(50):
   # index = np.random.randint(0,len(eqs))
   # eq = eqs[index]
   # size = np.random.uniform( 10, 14)
   # x,y = np.random.uniform(0,1,2)
   # alpha = np.random.uniform(0.1,0.58)
   # plt.text(x, y, eq, ha='center', va='center', color="#FF0000", alpha=alpha,
    #         transform=plt.gca().transAxes, fontsize=size, clip_on=True)

#plt.xticks([]), plt.yticks([])
#plt.imshow(np.squeeze(stylized_image))
#plt.show()

# 显示处理后的图像
for i in range(len(content_images)):
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(np.squeeze(content_images[i]))
    axs[1].imshow(np.squeeze(style_image))
    axs[2].imshow(np.squeeze(stylized_images[i]))
    
    for i in range(40):
        index = np.random.randint(0,len(a))
        eq = a[index]
        size = np.random.uniform( 7, 12)
        x,y = np.random.uniform(0,1,2)
        alpha = np.random.uniform(0.1,0.58)
        plt.text(x, y, eq, ha='center', va='center', color="#FFFFFF", alpha=alpha,
             transform=plt.gca().transAxes, fontsize=size, clip_on=True)
    
    # 关掉坐标轴显示
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
   
    plt.show()
    


# In[ ]:




