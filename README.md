## Keras Data Augmentation

This code aims to demonstrate the most recurrent methods when using data augmentation to increase the dataset of a convolutional neural network, using the python programming language and having as main reference the **keras** library and its **ImageDataGenerator** class.

##

### Basic information about the usage of the script

###### 1. Usage operation flow:

1. Install important packages with `!pip install tensorflow` and `!pip install opencv-python`:
2. Upload the desired images at the `dataset` folder, after created;
3. Run the notebook cells in order.

<br>

###### 2. ImageDataGenerator class:
Create and configure your ImageDataGenerator to fit it on your data.
```python
# Construindo uma instância da classe ImageDataGenerator:
imagens_geradas = ImageDataGenerator(
                rotation_range = 90,                           # Rotação entre 0 e 20°;
                brightness_range = [0.2, 1.2],                 # Variação do brilho entre 20% e 120%;
                channel_shift_range = 20,                      # Variação de saturação entre 0 e 20 unidades;
                zoom_range = [0.7, 1.0],                       # Variação do zoom entre 70% e 100%;
                vertical_flip = True,                          # Flip vertical;
                fill_mode = 'reflect')                         # Outros: nearest, constant, reflect, wrap.
```

<br>

###### 3. Flow function:
The data generator itself is, in fact, an iterator, returning batches of image samples when requested. You can configure the batch size and prepare the data generator and get batches of images by calling the **flow()** function.
```python
# Variável auxiliar:
aux = 0;

# Iterações do método flow, gerando 4 novas imagens para cada imagem normal:
for batch in imagens_geradas.flow(x, batch_size=16,  
                            save_to_dir = '/content/augmented', 
                            save_prefix = 'DJI_aug', 
                            save_format = 'jpg'):
    aux += 1;

    # Break:
    if aux == 4:
        break;
```

##

### Transformations and their results


###### 1. Vertical Flip:
Vertical flip is an image transformation technique used in data augmentation to create additional training examples by flipping images vertically, as shown below:
```python
imagens_geradas = ImageDataGenerator(vertical_flip = True);
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225726839-5dca7c16-c7de-441a-9afe-41c3ad89f653.png">
</p>

<br>

###### 2. Horizontal Flip:
Horizontal flip is a common image transformation used in data augmentation to create additional training examples by flipping images horizontally, as shown below:
```python
imagens_geradas = ImageDataGenerator(horizontal_flip = True);
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225727100-e21787c9-d9e1-48cb-ada8-3ca3f57c8f4b.png">
</p>

<br>

###### 3. Zoom:
Zoom in and zoom out are image transformation techniques used in data augmentation to create additional training examples by altering the scale of images, in this case it will vary between `70 and 100 %`, as shown below:
```python
imagens_geradas = ImageDataGenerator(zoom_range = [0.7, 1.0]);
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225727805-de2db3f3-3b30-4fb0-b022-c35547d91e65.png">
</p>

<br>

###### 4. Channel Shift:
Channel shift is an image transformation technique used in data augmentation to create additional training examples by shifting color channels, in this case it will vary them between `0 and 50`, as shown below:
```python
imagens_geradas = ImageDataGenerator(channel_shift_range = 50);
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225728083-fce74218-f75a-4164-86f3-4c40eaa6ab41.png">
</p>

<br>

###### 5. Brightness:
Brightness adjustment is an image transformation technique used in data augmentation to create additional training examples by altering the brightness of images, in this case it will vary between `120 and 130 %`, as shown below:
```python
imagens_geradas = ImageDataGenerator(brightness_range = [1.2, 1.3]);
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225728645-813e25d4-f58b-4589-9b7a-1a98f9a372e9.png">
</p>

<br>

###### 6. Rotation:
Rotation is an image transformation technique used in data augmentation to create additional training examples by rotating images, in this case it will vary between `0 and 20°`, as shown below:
```python
imagens_geradas = ImageDataGenerator(rotation_range = 20, fill_mode = 'reflect');
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225729043-72d8717c-53d5-4ce8-972c-b1c474b106bb.png">
</p>

<br>

###### 7. Width:
Width range is a data augmentation technique that randomly resizes images within a specified range of widths, in this case it will vary between `0 and 20 %`, as shown below:
```python
imagens_geradas = ImageDataGenerator(width_shift_range = 0.2, fill_mode = 'reflect');
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225729291-da97c532-ec46-4b9e-aa16-d1218f6f505b.png">
</p>

<br>

###### 8. Height:
Height range is a data augmentation technique that randomly resizes images within a specified range of heights, in this case it will vary between `0 and 20 %`, as shown below:
```python
imagens_geradas = ImageDataGenerator(height_shift_range = 0.2, fill_mode = 'reflect');
```

<p align="center">
    <img src="https://user-images.githubusercontent.com/96185134/225729475-b9666d52-6a0d-49af-8326-c53aba8ee977.png">
</p>

<br>

##

### Remider

Remember that this code can be used generically to fill your dataset but it is recommended that you visit the [keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) to get an idea of how to use data augmentation in flight time, while your convolutional neural network is being trained.

