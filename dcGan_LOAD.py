import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('/Users/a./Downloads/GAN MODEL',compile=False)
# model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5))

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input
def visualize_images(examples, n=7):
    examples = (examples+1)/2.0
    for i in range(n*n):
        plt.subplot(n,n,1+i)
        plt.axis('off')
        plt.imshow(examples[i])
    plt.show()

x = model.predict(generate_latent_points(100,50))
visualize_images(examples=x)
print(x.shape)
print(x)
