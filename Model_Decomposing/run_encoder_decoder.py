import tensorflow as tf
import sys
# lib_abs_path = 'C:\\Users\\sgkmccal\\Documents\\GitHub\\GP-VAE\\lib'
# sys.path.append(lib_abs_path)
# import lib
# from lib import nn_utils
import nn_utils
from nn_utils import make_nn, make_cnn, make_2d_cnn
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from encoder_isolated import DiagonalEncoder, BernoulliDecoder, JointEncoder, BandedJointEncoder 
from enum import Enum
from models import ImagePreprocessor, VAE, HI_VAE, GP_VAE
import time, datetime
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import os
from absl import app

#%% """ GENERIC PARAMS """
learning_rate = 1e-3
gradient_clip = 1e4
num_steps = 0
print_interval = 0
exp_name = "debug" # ???
basedir = 'models' 
# data_dir = "" overwritten/declared elsewhere, hardcoded to point to hmnist_mnar
data_type = ""
seed = 1337
debug = False

# Define output folder, produce experiment name

timestamp = datetime.now().strftime("%y%m%d")
full_exp_name = "{}_{}".format(timestamp, exp_name)
outdir = basedir + full_exp_name

# model_type = Enum('gp-vae', ['vae', 'hi-vae', 'gp-vae'])
model_type = '' # !!!: Change accordingly

cnn_kernel_size = 3
cnn_sizes = [256]
testing = False
banded_covar = False
batch_size = 64
M=1 # Num. samples ELBO estimation
K=1 # Num. importance sampling
kernel = Enum('kernel', 'cauchy', ['rbf', 'diffusion', 'matern', 'cauchy'])
kernel_scales = 1

#%%""" ------ HMNIST PARAMS ------"""
# z: dims. of latent space
# for images, each pixel is initially treated as its own dimension/feature
# e.g. for 64x64 px image, image has 64^2 = 4096 dims
# so z should be less than this to compress, but how much smaller
# depends on goal of encoding i.e. smallest size, preserving most
# detail etc.
z_size = 256 # paper set z=256 for hmnist data

data_dir = "data/hmnist/hmnist_mnar.npz" # hardcoded to point to hmnist_manr
data_dim = 784
time_length = 10
num_classes = 10
decoder = BernoulliDecoder
img_shape = (28, 28, 1)
val_split = 50000

latent_dim = 256
encoder_sizes = [32,256,256] 
decoder_sizes = [256,256,256]
window_size = 3
sigma = 1.0
length_scale = 2.0
beta = 0.1
num_epochs = 20 


data = np.load('./data/hmnist/hmnist_mnar.npz')


def main(argv):
    del argv
    #%% HMNIST DATA
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    print("Testing: ", testing, f"\t Seed: {seed}")

    encoder_sizes = [int(size) for size in encoder_sizes]
    decoder_sizes = [int(size) for size in decoder_sizes]

    if 0 in encoder_sizes:
        encoder_sizes.remove(0)
    if 0 in decoder_sizes:
        decoder_sizes.remove(0)

    # Make up full exp name
    timestamp = datetime.now().strftime("%y%m%d")
    full_exp_name = "{}_{}".format(timestamp, exp_name)
    outdir = os.path.join(basedir, full_exp_name)
    if not os.path.exists(outdir): os.mkdir(outdir)
    checkpoint_prefix = os.path.join(outdir, "ckpt")
    print("Full exp name: ", full_exp_name)

    x_train_full = data['x_train_full']
    x_train_miss = data['x_train_miss']
    m_train_miss = data['m_train_miss']
    y_train = data['y_train']

    # Testing data
    # x_val_full = data['x_test_full']
    # x_val_miss = data['x_test_miss']
    # m_val_miss = data['m_test_miss']
    # y_val = data['y_test']

    # 
    x_val_full = x_train_full[val_split:]
    x_val_miss = x_train_miss[val_split:]
    m_val_miss = m_train_miss[val_split:]
    y_val = y_train[val_split:]
    x_train_full = x_train_full[:val_split]
    x_train_miss = x_train_miss[:val_split]
    m_train_miss = m_train_miss[:val_split]
    y_train = y_train[:val_split]

    # refactor some data into Tensorflow object (?)
    tf_x_train_miss = tf.data.Dataset \
                        .from_tensor_slices((x_train_miss, m_train_miss))\
                        .shuffle(len(x_train_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)) \
                        .batch(batch_size).repeat()
    tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    # Build Conv2D preprocessor for image data
    print("HMNIST dataset, so using CNN preprocessor")
    image_preprocessor = ImagePreprocessor(img_shape,    \
                                        cnn_sizes,    \
                                        cnn_kernel_size)

    # %% Model initialisation
    if model_type == "vae":
        model = VAE(latent_dim=latent_dim, data_dim=data_dim, time_length=time_length,
                    encoder_sizes=encoder_sizes, encoder=DiagonalEncoder,
                    decoder_sizes=decoder_sizes, decoder=decoder,
                    image_preprocessor=image_preprocessor, window_size=window_size,
                    beta=beta, M=M, K=K)
    elif model_type == "hi-vae":
        model = HI_VAE(latent_dim=latent_dim, data_dim=data_dim, time_length=time_length,
                        encoder_sizes=encoder_sizes, encoder=DiagonalEncoder,
                        decoder_sizes=decoder_sizes, decoder=decoder,
                        image_preprocessor=image_preprocessor, window_size=window_size,
                        beta=beta, M=M, K=K)
    elif model_type == "gp-vae":
        encoder = BandedJointEncoder if banded_covar else JointEncoder
        model = GP_VAE(latent_dim=latent_dim, data_dim=data_dim, time_length=time_length,
                        encoder_sizes=encoder_sizes, encoder=encoder,
                        decoder_sizes=decoder_sizes, decoder=decoder,
                        kernel=kernel, sigma=sigma,
                        length_scale=length_scale, kernel_scales = kernel_scales,
                        image_preprocessor=image_preprocessor, window_size=window_size,
                        beta=beta, M=M, K=K, data_type=data_type)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")

    # %% Training preparation
    print("GPU support: ", tf.test.is_gpu_available())

    print("Training...")
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())

    if model.preprocessor is not None:
        print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net,
                                                decoder=model.decoder.net, preprocessor=model.preprocessor.net,
                                                optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    else:
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, decoder=model.decoder.net,
                                                optimizer_step=tf.compat.v1.train.get_or_create_global_step())

    summary_writer = tf.contrib.summary.create_file_writer(outdir, flush_millis=10000)

    if num_steps == 0:
        num_steps = num_epochs * len(x_train_miss) // batch_size
    else:
        num_steps = num_steps

    if print_interval == 0:
        print_interval = num_steps // num_epochs

    #%% Training
    losses_train = []
    losses_val = []

    t0 = time.time()
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for i, (x_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):
            try:
                with tf.GradientTape() as tape:
                    tape.watch(trainable_vars)
                    loss = model.compute_loss(x_seq, m_mask=m_seq)
                    losses_train.append(loss.numpy())
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
                optimizer.apply_gradients(zip(grads, trainable_vars),
                                            global_step=tf.compat.v1.train.get_or_create_global_step())

                # Print intermediate results
                if i % print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer._lr, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    saver.save(checkpoint_prefix)
                    tf.contrib.summary.scalar("loss_train", loss)
                    tf.contrib.summary.scalar("kl_train", kl)
                    tf.contrib.summary.scalar("nll_train", nll)

                    # Validation loss
                    x_val_batch, m_val_batch = tf_x_val_miss.get_next()
                    val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(val_loss, val_nll, val_kl))

                    tf.contrib.summary.scalar("loss_val", val_loss)
                    tf.contrib.summary.scalar("kl_val", val_kl)
                    tf.contrib.summary.scalar("nll_val", val_nll)

                    if data_type in ["hmnist", "sprites"]:
                        # Draw reconstructed images
                        x_hat = model.decode(model.encode(x_seq).sample()).mean()
                        tf.contrib.summary.image("input_train", tf.reshape(x_seq, [-1]+list(img_shape)))
                        tf.contrib.summary.image("reconstruction_train", tf.reshape(x_hat, [-1]+list(img_shape)))
                    elif data_type == 'physionet':
                        # Eval MSE and AUROC on entire val set
                        # x_val_miss_batches = np.array_split(x_val_miss, batch_size, axis=0)
                        # x_val_full_batches = np.array_split(x_val_full, batch_size, axis=0)
                        # m_val_artificial_batches = np.array_split(m_val_artificial, batch_size, axis=0)
                        # get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_artificial_batches)

                        # n_missings = m_val_artificial.sum()
                        # mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m).numpy()
                        #                     for x, y, m in get_val_batches()]) / n_missings

                        # x_val_imputed = np.vstack([model.decode(model.encode(x_batch).mean()).mean().numpy()
                        #                             for x_batch in x_val_miss_batches])
                        # x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]  # impute gt observed values

                        # x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])
                        # val_split = len(x_val_imputed) // 2
                        # cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
                        # cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
                        # probs = cls_model.predict_proba(x_val_imputed[val_split:])[:, 1]
                        # auroc = roc_auc_score(y_val[val_split:], probs)
                        # print("MSE miss: {:.4f} | AUROC: {:.4f}".format(mse_miss, auroc))

                        # # Update learning rate (used only for physionet with decay=0.5)
                        # if i > 0 and i % (10*print_interval) == 0:
                        #     optimizer._lr = max(0.5 * optimizer._lr, 0.1 * learning_rate)
                        pass

                    t0 = time.time()
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                if debug:
                    import ipdb
                    ipdb.set_trace()
                break


    #%% Evaluation
    # Evaluation #
        ##############

    print("Evaluation...")

    # Split data on batches
    x_val_miss_batches = np.array_split(x_val_miss, batch_size, axis=0)
    x_val_full_batches = np.array_split(x_val_full, batch_size, axis=0)

    if data_type == 'physionet':
        # m_val_batches = np.array_split(m_val_artificial, batch_size, axis=0)
        pass
    else:
        m_val_batches = np.array_split(m_val_miss, batch_size, axis=0)

    get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_batches)

    # Compute NLL and MSE on missing values
    # n_missings = m_val_artificial.sum() if data_type == 'physionet' else m_val_miss.sum()
    n_missings = m_val_miss.sum()
    nll_miss = np.sum([model.compute_nll(x, y=y, m_mask=m).numpy()
                        for x, y, m in get_val_batches()]) / n_missings
    mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m, binary=data_type=="hmnist").numpy()
                        for x, y, m in get_val_batches()]) / n_missings
    print("NLL miss: {:.4f}".format(nll_miss))
    print("MSE miss: {:.4f}".format(mse_miss))

    # Save imputed values
    z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
    np.save(os.path.join(outdir, "z_mean"), np.vstack(z_mean))
    x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
    np.save(os.path.join(outdir, "imputed_no_gt"), x_val_imputed)

    # impute gt observed values
    x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
    np.save(os.path.join(outdir, "imputed"), x_val_imputed)

    # if data_type == "hmnist":
    # AUROC evaluation using Logistic Regression
    x_val_imputed = np.round(x_val_imputed)
    x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])

    cls_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
    val_split = len(x_val_imputed) // 2

    cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
    probs = cls_model.predict_proba(x_val_imputed[val_split:])

    auprc = average_precision_score(np.eye(num_classes)[y_val[val_split:]], probs)
    auroc = roc_auc_score(np.eye(num_classes)[y_val[val_split:]], probs)
    print("AUROC: {:.4f}".format(auroc))
    print("AUPRC: {:.4f}".format(auprc))

    #%% Visualize reconstructions
    # if data_type in ["hmnist", "sprites"]:
    img_index = 0
    if data_type == "hmnist":
        img_shape = (28, 28)
        cmap = "gray"
    elif data_type == "sprites":
        img_shape = (64, 64, 3)
        cmap = None

    fig, axes = plt.subplots(nrows=3, ncols=x_val_miss.shape[1], figsize=(2*x_val_miss.shape[1], 6))

    x_hat = model.decode(model.encode(x_val_miss[img_index: img_index+1]).mean()).mean().numpy()
    seqs = [x_val_miss[img_index:img_index+1], x_hat, x_val_full[img_index:img_index+1]]

    for axs, seq in zip(axes, seqs):
        for ax, img in zip(axs, seq[0]):
            ax.imshow(img.reshape(img_shape), cmap=cmap)
            ax.axis('off')

    suptitle = model_type + f" reconstruction, NLL missing = {mse_miss}"
    fig.suptitle(suptitle, size=18)
    fig.savefig(os.path.join(outdir, data_type + "_reconstruction.pdf"))

    results_all = [seed, model_type, data_type, kernel, beta, latent_dim,
                    num_epochs, batch_size, learning_rate, window_size,
                    kernel_scales, sigma, length_scale,
                    len(encoder_sizes), encoder_sizes[0] if len(encoder_sizes) > 0 else 0,
                    len(decoder_sizes), decoder_sizes[0] if len(decoder_sizes) > 0 else 0,
                    cnn_kernel_size, cnn_sizes,
                    nll_miss, mse_miss, losses_train[-1], losses_val[-1], auprc, auroc, testing, data_dir]

    with open(os.path.join(outdir, "results.tsv"), "w") as outfile:
        outfile.write("seed\tmodel\tdata\tkernel\tbeta\tz_size\tnum_epochs"
                        "\tbatch_size\tlearning_rate\twindow_size\tkernel_scales\t"
                        "sigma\tlength_scale\tencoder_depth\tencoder_width\t"
                        "decoder_depth\tdecoder_width\tcnn_kernel_size\t"
                        "cnn_sizes\tNLL\tMSE\tlast_train_loss\tlast_val_loss\tAUPRC\tAUROC\ttesting\tdata_dir\n")
        outfile.write("\t".join(map(str, results_all)))

    with open(os.path.join(outdir, "training_curve.tsv"), "w") as outfile:
        outfile.write("\t".join(map(str, losses_train)))
        outfile.write("\n")
        outfile.write("\t".join(map(str, losses_val)))

    print("Training finished.")


if __name__ == '__main__':
    app.run(main)

#%% """ GP-VAE MODEL """
"""
Banded Joint Encoder
Encoder used for GP VAE model w/ proposed banded cov. matrix
z_size: latent space dimensionality
       = num. dimensions in compressed representation of input data
         that the encoder produces
       i.e. output of encoder has {z} dimensions 
hidden_sizes: num. of neurons in hidden layers of encoder
              for 64x64 px image, input_size = 64*64 = 4096
              hidden layers sizes should be smaller than this to
              gradually lower input size towards latent size
window_size: kernel size for conv1D layer
"""
# bje_enc = BandedJointEncoder(z_size=z_size) 


