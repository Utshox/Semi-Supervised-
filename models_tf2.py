import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
class UNetBlock(layers.Layer):
    """
    Basic U-Net convolutional block with batch normalization
    """
class UNetBlock(layers.Layer):
    """Basic U-Net convolutional block with batch normalization"""
    def __init__(self, filters, name='unet_block', **kwargs):
        super(UNetBlock, self).__init__(name=name, **kwargs)
        self.filters = filters

        # First convolution
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv1'
        )
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn1')

        # Second convolution
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv2'
        )
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn2')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

class PancreasSeg(Model):
    """
    U-Net model for pancreas segmentation with support for semi-supervised learning
    """
    def __init__(self, config):
          super(PancreasSeg, self).__init__()
          
          self.img_size = (config.img_size_x, config.img_size_y)
          self.num_classes = config.num_classes
          n_filters = config.n_filters # Use base filter number from config

          # Encoder blocks with increasing filters based on config.n_filters
          self.enc1 = UNetBlock(filters=n_filters, name='enc1') # e.g., 32
          self.enc2 = UNetBlock(filters=n_filters*2, name='enc2') # e.g., 64
          self.enc3 = UNetBlock(filters=n_filters*4, name='enc3') # e.g., 128
          self.enc4 = UNetBlock(filters=n_filters*8, name='enc4') # e.g., 256

          # Bridge
          self.bridge = UNetBlock(filters=n_filters*16, name='bridge') # e.g., 512

          # Decoder blocks with decreasing filters
          self.dec4 = UNetBlock(filters=n_filters*8, name='dec4') # e.g., 256
          self.dec3 = UNetBlock(filters=n_filters*4, name='dec3') # e.g., 128
          self.dec2 = UNetBlock(filters=n_filters*2, name='dec2') # e.g., 64
          self.dec1 = UNetBlock(filters=n_filters, name='dec1') # e.g., 32

          # Upsampling layers - Replace UpSampling2D with Conv2DTranspose
          self.up4 = layers.Conv2DTranspose(filters=n_filters*16, kernel_size=2, strides=2, padding='same', name='up4_trans') # Input 512 -> Output 512 channels, 2x size
          self.up3 = layers.Conv2DTranspose(filters=n_filters*8, kernel_size=2, strides=2, padding='same', name='up3_trans')  # Input 256 -> Output 256 channels, 2x size
          self.up2 = layers.Conv2DTranspose(filters=n_filters*4, kernel_size=2, strides=2, padding='same', name='up2_trans')  # Input 128 -> Output 128 channels, 2x size
          self.up1 = layers.Conv2DTranspose(filters=n_filters*2, kernel_size=2, strides=2, padding='same', name='up1_trans')  # Input 64 -> Output 64 channels, 2x size

          # Max pooling layers
          self.pool = layers.MaxPooling2D(pool_size=(2, 2))

          # Final convolution
          self.final_conv = layers.Conv2D(
              filters=self.num_classes,
              kernel_size=1,
              activation=None, # Output logits
              kernel_initializer=tf.keras.initializers.HeNormal(),
              bias_initializer='zeros' # Changed from Constant(-2.19)
          )
          
          # Initialize with a dummy input to create weights
          self.built = False
    def build(self, input_shape):
            """Initialize the model with a specific input shape"""
            super().build(input_shape)
            dummy_input = tf.zeros((1,) + input_shape[1:])
            _ = self.call(dummy_input, training=False)
            print(f"Model built with input shape: {input_shape}")

    def call(self, inputs, training=False):
        """Forward pass of the model"""
        # Ensure input is float32
        x = tf.cast(inputs, tf.float32)
        
        # Encoder Path
        e1 = self.enc1(x, training=training)
        p1 = self.pool(e1)

        e2 = self.enc2(p1, training=training)
        p2 = self.pool(e2)

        e3 = self.enc3(p2, training=training)
        p3 = self.pool(e3)

        e4 = self.enc4(p3, training=training)
        p4 = self.pool(e4)

        # Bridge
        b = self.bridge(p4, training=training)

        # Decoder Path
        u4 = self.up4(b)
        c4 = tf.concat([u4, e4], axis=-1)
        d4 = self.dec4(c4, training=training)

        u3 = self.up3(d4)
        c3 = tf.concat([u3, e3], axis=-1)
        d3 = self.dec3(c3, training=training)

        u2 = self.up2(d3)
        c2 = tf.concat([u2, e2], axis=-1)
        d2 = self.dec2(c2, training=training)

        u1 = self.up1(d2)
        c1 = tf.concat([u1, e1], axis=-1)
        d1 = self.dec1(c1, training=training)

        # Final convolution (no activation)
        logits = self.final_conv(d1)

        return logits
      


    def get_projections(self, x, training=False):
        """Get embeddings for contrastive learning"""
        bridge, _ = self.encode(x, training=training)
        projections = self.projection(bridge, training=training)
        return projections


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, config, smooth=1e-6, alpha = 0.25, gamma = 2):
        super().__init__()
        self.config = config
        self.smooth = smooth
        self.alpha = alpha 
        self.gamma = gamma

    def dice_loss(self, y_true, y_pred):
          y_true = tf.cast(y_true, tf.float32)
          y_pred = tf.cast(y_pred, tf.float32)
          
          # Apply sigmoid here
          y_pred = tf.nn.sigmoid(y_pred)
          
          # Simple Dice implementation
          intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
          union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
          
          dice = (2. * intersection + self.smooth) / (union + self.smooth)
          return 1.0 - tf.reduce_mean(dice)

    def focal_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute class weights dynamically
        pos_weight = 1.0 / (tf.reduce_mean(y_true) + 1e-7)
        neg_weight = 1.0 / (tf.reduce_mean(1 - y_true) + 1e-7)
        
        # Calculate focal weights
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), pos_weight, neg_weight)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Compute binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(focal_weight * bce)
    def call(self, y_true, y_pred):
        # Take only pancreas class
        y_true_fg = y_true[..., 1:]
        y_pred_fg = y_pred[..., 1:]
        
        # Calculate only Dice loss initially
        dice_loss = self.dice_loss(y_true_fg, y_pred_fg)
        
        return dice_loss + self.focal_loss(y_true_fg, y_pred_fg)
        #return dice_loss

       
class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Contrastive loss for self-supervised learning
    """
    def __init__(self, temperature=0.1, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, projection_1, projection_2):
        # Normalize projections
        proj_1 = tf.math.l2_normalize(projection_1, axis=1)
        proj_2 = tf.math.l2_normalize(projection_2, axis=1)

        # Compute similarity matrix
        batch_size = tf.shape(proj_1)[0]
        similarity_matrix = tf.matmul(proj_1, proj_2, transpose_b=True) / self.temperature

        # Positive pairs are on the diagonal
        positives = tf.linalg.diag_part(similarity_matrix)

        # All similarities act as negatives for each positive
        negatives = tf.reshape(similarity_matrix, [-1])

        # Create labels: positives = 1, negatives = 0
        labels = tf.eye(batch_size)

        # Compute cross entropy loss
        loss = tf.keras.losses.categorical_crossentropy(
            labels,
            tf.nn.softmax(similarity_matrix),
            from_logits=False
        )

        return tf.reduce_mean(loss)
def create_ssl_model(config):
    """
    Factory function to create model with losses and optimizers
    """
    model = PancreasSeg(config)
    
    # Ensure model is initialized
    input_shape = (None, config.img_size_x, config.img_size_y, config.num_channels)
    model.build(input_shape)
    
    # Define optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    # Define losses
    supervised_loss = DiceLoss()

    return {
        'model': model,
        'optimizer': optimizer,
        'supervised_loss': supervised_loss,
    }