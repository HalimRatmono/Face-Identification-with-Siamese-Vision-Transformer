import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from transformers import TFViTModel
import modules.distances as distances

class SiameseNetwork(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
     
        self.pixel_values1 = tf.keras.layers.Input(shape=(224,224, 3), name='pixel_values1', dtype='float32')
        self.pixel_values2 = tf.keras.layers.Input(shape=(224,224, 3), name='pixel_values2', dtype='float32')

        self.p1 = layers.Permute((3, 1, 2), input_shape=(224,224, 3))
        self.p2 = layers.Permute((3, 1, 2), input_shape=(244,224, 3))
    
        self.vit1 = self.base_model.vit
        self.vit2 = self.base_model.vit
    
        self.distances = distances.L1Dist()
        
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')

        # model
        # keras_model = tf.keras.Model(inputs=[pixel_values1,pixel_values2] , outputs=classifier,name="siamese_network")

        
    def make_siamese_net(self):

        p1 = self.p1(self.pixel_values1)
        p2 = self.p2(self.pixel_values2)
    
        vit1 = self.vit1(p1)[1]
        vit2 = self.vit2(p2)[1]

        distances = self.distances(vit1, vit2)

        classifier = self.classifier(distances)

        # model
        keras_model = tf.keras.Model(inputs=[self.pixel_values1,self.pixel_values2] , outputs=classifier,name="siamese_network")

        return keras_model



    # def call(self, input):

    #     input1, input2 = input
    #     print("INPUT1")
    #     print(type(input1))
    #     print(input1.shape)
    #     print("INPUT2")
    #     print(input2.shape)
    #     pixel_values1 = self.pixel_values1(input1)
    #     pixel_values2 = self.pixel_values2(input2)
    #     p1 = self.p1(pixel_values1)
    #     p2 = self.p2(pixel_values2)
    #     vit1 = self.vit1(p1)
    #     vit2 = self.vit2(p2)
    #     distances = self.distances(vit1[0],vit2[0])
        
    #     return self.classifier(distances)

