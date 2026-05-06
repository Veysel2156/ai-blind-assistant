#Hücre 1////////////////////////////////////////
import os
import cv2
import yaml
import glob

INPUT_DIR = '/kaggle/input/datasets/veyselbaran/householdv3'
OUTPUT_DIR = '/kaggle/working/keras_dataset'

yaml_path = os.path.join(INPUT_DIR, 'data.yaml')
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
    
class_names = data['names']
if isinstance(class_names, list):
    class_names = {i: name for i, name in enumerate(class_names)}

print(f"Tespit edilen sınıflar: {class_names}")

# DÜZELTME: 'test' klasörü de eklendi!
for split in ['train', 'valid', 'test']:
    for class_id, class_name in class_names.items():
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

def process_yolo_to_keras(split):
    img_paths = glob.glob(f'{INPUT_DIR}/{split}/**/*.jpg', recursive=True)
    img_paths += glob.glob(f'{INPUT_DIR}/{split}/**/*.png', recursive=True)
    
    count = 0
    for img_path in img_paths:
        txt_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(txt_path):
            txt_path = img_path.rsplit('.', 1)[0] + '.txt'
            
        if os.path.exists(txt_path):
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    x1 = max(0, int((x_center - width/2) * w))
                    y1 = max(0, int((y_center - height/2) * h))
                    x2 = min(w, int((x_center + width/2) * w))
                    y2 = min(h, int((y_center + height/2) * h))
                    
                    cropped_img = img[y1:y2, x1:x2]
                    
                    if cropped_img.size > 0:
                        class_name = class_names[class_id]
                        save_path = os.path.join(OUTPUT_DIR, split, class_name, f"img_{count}.jpg")
                        cv2.imwrite(save_path, cropped_img)
                        count += 1
                        
    print(f"{split.upper()} aşaması bitti: {count} adet nesne ayrıştırıldı.")

print("Kırpma ve Klasörleme başlıyor...")
process_yolo_to_keras('train')
process_yolo_to_keras('valid')
process_yolo_to_keras('test') # DÜZELTME: Test verileri de işleniyor!
print("Veri hazırlığı tamamlandı!")


#Hücre 2 //////////////////////////////////////////////////////////////////
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

TRAIN_DIR = '/kaggle/working/keras_dataset/train'
VALID_DIR = '/kaggle/working/keras_dataset/valid'
TEST_DIR = '/kaggle/working/keras_dataset/test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Eğitim veri seti yükleniyor...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

print("Doğrulama veri seti yükleniyor...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

NUM_CLASSES = len(train_dataset.class_names)
print(f"\nSınıflar ({NUM_CLASSES} adet): {train_dataset.class_names}")

# --- SINIF AĞIRLIKLARI (CLASS WEIGHTS) HESAPLANIYOR ---
print("\nSınıf Ağırlıkları Hesaplanıyor (Az fotoğraflı nesnelere daha çok önem verilecek)...")
y_train = np.concatenate([y for x, y in train_dataset], axis=0)
y_train_classes = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_classes),
    y=y_train_classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Sınıf Ağırlıkları: {class_weight_dict}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


#Hücre 3 ///////////////////////////////////////////////////////////////////////////////////////
# GELİŞMİŞ VERİ ARTIRMA: Parlaklık, kontrast ve kaydırma eklendi
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomContrast(factor=0.2),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x) # EZBERİ BOZMAK İÇİN 0.3'TEN 0.4'E ÇIKARILDI
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

#Hücre 4 /////////////////////////////////////////////////////////////////////////////////////////////
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
)

callbacks_list = [early_stopping, reduce_lr]


#Hücre 5 ///////////////////////////////////////////////////////////////////////////////////
print("--- Aşama 1: Sınıflandırıcı Eğitimi Başlıyor ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50, 
    callbacks=callbacks_list,
    class_weight=class_weight_dict # Sınıf ağırlıkları devrede!
)


#Hücre 6/////////////////////////////////////////////////////////////////////////////////////
print("--- Aşama 2: Fine-Tuning Başlıyor ---")
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=150,  # 150 Epoch! (Erken durdurma devrede)
    callbacks=callbacks_list,
    class_weight=class_weight_dict # Sınıf ağırlıkları burada da devrede!
)


#Hücre 7//////////////////////////////////////////////////////////////////////////////////////
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. GRAFİKLERİ ÇİZ (Burası aynı, eğitim sürecini gösterir)
acc = history_2.history['accuracy']
val_acc = history_2.history['val_accuracy']
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Eğitim Başarısı')
plt.plot(epochs_range, val_acc, 'r-', linewidth=2, label='Doğrulama Başarısı')
plt.title('Model Başarısı (Accuracy)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, 'r-', linewidth=2, label='Doğrulama Kaybı')
plt.title('Model Hata (Loss)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 2. CONFUSION MATRIX (KARMAŞIKLIK MATRİSİ) - ARTIK TEST VERİSİ İLE!
print("\nGerçek Sınav (TEST) üzerinden Confusion Matrix hesaplanıyor...")
# DÜZELTME: VALID_DIR yerine TEST_DIR kullanılıyor!
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False 
)
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
plt.title('Gerçek Sınav Sonucu: Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('GERÇEK Nesne')
plt.xlabel('TAHMİN EDİLEN Nesne')
plt.xticks(rotation=45, ha='right')
plt.show()

print("\n--- GERÇEK SINAV: DETAYLI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=test_dataset.class_names))



#Hücre 8/////////////////////////////////////////////////////////////////////////////////////////////////////
print("--- Model Gerçek int8 TFLite Formatına Dönüştürülüyor ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    # Modelin renkleri 0-255 arasından -128 ile +127 arasına nasıl çekeceğini öğrenmesi için
    for input_value, _ in train_dataset.unbatch().batch(1).take(100):
        yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# DÜZELTME: uint8 yerine gerçek int8 (-128, +127) standardına çekildi!
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 

tflite_quant_model = converter.convert()

tflite_path = 'model_optimized_int8.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Mükemmel! Model başarıyla tam int8 standardında optimize edildi ve '{tflite_path}' olarak kaydedildi.")
