import tensorflow as tf
import numpy as np
import pathlib

batch_size = 4
img_height = 50
img_width = 26
epochs = 20
period = 20
data_path = pathlib.Path('project\data_num')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  labels="inferred",
  label_mode="categorical",
  image_size=(img_height, img_width),
  batch_size=batch_size)

converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/floatGray/Desktop/project/weights/model/')

tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/floatGray/Desktop/project/weights/model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()
for element in train_ds:
    e=element

img_tensor = tf.constant(e[0])

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(img_tensor).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/floatGray/Desktop/project/weights/model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()



interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)


tflite_models_dir = pathlib.Path("C:/Users/floatGray/Desktop/project/weights/model/quant/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# 保存量化后的模型:
tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)