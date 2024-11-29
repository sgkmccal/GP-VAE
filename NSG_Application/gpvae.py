import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

"""
GP-VAE implementation with additional classification network
Classification network takes latent rep. z as input, outputs binary classification 
Loss now sum of ELBO and classification loss (BCE(???))
L_total = L_VAE + aL_class
a again trade-off coeff. between two components
"""


# TensorFlow Probability aliases
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

class GPVAEClassifier(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, kernel_variance=1.0, kernel_lengthscale=1.0):
        super(GPVAEClassifier, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: Maps input to latent space parameters
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2 * latent_dim)  # Mean and log-variance output
        ])

        # GP kernel
        self.kernel = tfk.ExponentiatedQuadratic(variance=kernel_variance, length_scale=kernel_lengthscale)

        # Decoder: Maps latent space to reconstructed input
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim)  # Reconstructed input
        ])

        # Classification Head: Maps latent space to class probabilities
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
        ])

    def call(self, inputs, training=False):
        # Encoder output
        encoder_output = self.encoder(inputs)
        mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        std_dev = tf.exp(0.5 * log_var)
        q_z = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std_dev)

        # GP prior
        gp_prior = tfd.GaussianProcess(kernel=self.kernel, index_points=tf.range(self.latent_dim, dtype=tf.float32))

        # Sample from latent space
        z = q_z.sample()

        # Decoder output (reconstruction)
        reconstruction = self.decoder(z)

        # Classification output
        class_probs = self.classifier(z)

        return reconstruction, class_probs, q_z, gp_prior

    def compute_loss(self, x, y, alpha=1.0):
        reconstruction, class_probs, q_z, gp_prior = self(x, training=True)

        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))

        # KL divergence
        kl_loss = tf.reduce_mean(tfd.kl_divergence(q_z, gp_prior))

        # Classification loss (Binary Cross-Entropy)
        classification_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, class_probs))

        # Total loss
        total_loss = reconstruction_loss + kl_loss + alpha * classification_loss
        return total_loss, reconstruction_loss, kl_loss, classification_loss

# Training the GP-VAE Classifier
def train_gpvae_classifier(model, dataset, optimizer, epochs=50, alpha=1.0):
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                loss, rec_loss, kl_loss, cls_loss = model.compute_loss(batch_x, batch_y, alpha)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss

        print(f"Epoch {epoch + 1}, Loss: {total_loss.numpy()}, Rec Loss: {rec_loss.numpy()}, KL Loss: {kl_loss.numpy()}, Cls Loss: {cls_loss.numpy()}")

# Sample Data and Training
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    x_data = np.random.rand(1000, 50).astype("float32")  # 50 input features
    y_data = (np.sum(x_data, axis=1) > 25).astype("float32")  # Binary labels: 1 if sum > threshold

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(64)

    # Initialize GP-VAE Classifier
    latent_dim = 10
    input_dim = 50
    model = GPVAEClassifier(input_dim=input_dim, latent_dim=latent_dim)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Train model
    train_gpvae_classifier(model, dataset, optimizer, epochs=20, alpha=1.0)


