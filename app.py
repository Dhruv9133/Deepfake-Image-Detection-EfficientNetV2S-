#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


# In[2]:


train_data_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Train'
valid_data_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Validation'
test_data_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Test'


# In[3]:


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[4]:


train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size=(224, 224),
                                                    batch_size=32, class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(224, 224),
                                                  batch_size=32, class_mode='binary', shuffle=False)


# In[5]:


base_model = tf.keras.applications.EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# In[6]:


model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.00005), metrics=['accuracy'])


# In[7]:


history = model.fit(train_generator, epochs=5, validation_data=valid_generator)


# In[17]:


test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")test_generator.reset()
Y_pred = model.predict(test_generator)
Y_pred = (Y_pred > 0.5).astype(int)
Y_true = test_generator.classes

conf_matrix = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(Y_true, Y_pred, target_names=['Fake', 'Real']))


# In[ ]:


test_generator.reset()
Y_pred = model.predict(test_generator)
Y_pred = (Y_pred > 0.5).astype(int)
Y_true = test_generator.classes

conf_matrix = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(Y_true, Y_pred, target_names=['Fake', 'Real']))


# In[20]:


y_true = test_generator.classes
y_pred = model.predict(test_generator)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()


# In[21]:


def compute_psnr(original_img, modified_img):
    mse = np.mean((original_img - modified_img) ** 2)
    if mse == 0:
        return 100  # No noise
    max_pixel = 1.0  # Since images are normalized
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# In[22]:


def generate_gradcam(model, img_array, layer_name="top_conv"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


# In[23]:


def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img


# In[24]:


model.save('deepfake_detector.h5')


# In[25]:


from tensorflow.keras.preprocessing import image

def test_single_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = 'Real' if prediction > 0.5 else 'Fake'
    print(f"Predicted Label: {label} (Confidence: {prediction:.2f})")


# In[26]:


image_path = "/kaggle/input/deepfake-and-real-images/Dataset/Test/Fake/fake_10.jpg"
test_single_image(image_path)

