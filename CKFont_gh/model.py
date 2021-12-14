import tensorflow as tf
import collections
import os
import glob

from ops import *

EPS = 1e-12

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
tgt_font_path = os.path.join(SCRIPT_PATH, 'tgt_font')

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, \
        disc_real_loss, disc_fake_loss, disc_loss_real_styl, discrim_grads_and_vars, \
        gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

# parameters for style embedding
train_num_styles = len(glob.glob1(tgt_font_path,"*.ttf"))
fine_tune_styles = 17
total_styles = train_num_styles + fine_tune_styles

total_characters = 2000

##################################################################################
# Content Encoder
##################################################################################
def create_content_enc(content_img, a):
    layers = []

    print('Content image shape is ', content_img.shape)
    print()
    channel = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("content_encoder_1"):
        x1 = conv(content_img, channel, kernel=7, stride=1, pad=3, pad_type='reflect')
        x1 = instance_norm(x1)
        x1 = tf.nn.relu(x1)
        layers.append(x1)
        print('Content encoder 1 shape is ', x1.shape)
        print()

    with tf.variable_scope("content_encoder_2"):
        x2 = conv(x1, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect')
        x2 = instance_norm(x2)
        x2 = tf.nn.relu(x2)
        layers.append(x2)
        print('Content encoder 2 shape is ', x2.shape)
        print()

    with tf.variable_scope("content_encoder_3"):
        x3 = conv(x2, channel * 4, kernel=4, stride=2, pad=1, pad_type='reflect')
        x3 = instance_norm(x3)
        x3 = tf.nn.relu(x3)
        layers.append(x3)
        print('Content encoder 3 shape is ', x3.shape)
        print()

    with tf.variable_scope("content_encoder_4"):
        x4 = conv(x3, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x4 = instance_norm(x4)
        x4 = tf.nn.relu(x4)
        layers.append(x4)
        print('Content encoder 4 shape is ', x4.shape)
        print()

    with tf.variable_scope("content_encoder_5"):
        x5 = conv(x4, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x5 = instance_norm(x5)
        x5 = tf.nn.relu(x5)
        layers.append(x5)
        print('Content encoder 5 shape is ', x5.shape)
        print()

    with tf.variable_scope("content_encoder_6"):
        x6 = conv(x5, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x6 = instance_norm(x6)
        x6 = tf.nn.relu(x6)
        layers.append(x6)
        print('Content encoder 6 shape is ', x6.shape)
        print()

    return x6, layers

##################################################################################
# Style Encoder
##################################################################################
def create_style_enc(src_1stSpt, src_2ndSpt, src_3rdSpt, a):
    layers = []
    style_imgs = tf.concat([src_1stSpt, src_2ndSpt, src_3rdSpt], axis=3)
    print('Style image shape is ', style_imgs.shape)
    print()
    channel = a.ngf
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("style_encoder_1"):
        x1 = conv(style_imgs, channel, kernel=7, stride=1, pad=3, pad_type='reflect')
        x1 = instance_norm(x1)
        x1 = tf.nn.relu(x1)
        layers.append(x1)
        print('Style encoder 1 shape is ', x1.shape)
        print()

    with tf.variable_scope("style_encoder_2"):
        x2 = conv(x1, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect')
        x2 = instance_norm(x2)
        x2 = tf.nn.relu(x2)
        layers.append(x2)
        print('Style encoder 2 shape is ', x2.shape)
        print()

    with tf.variable_scope("style_encoder_3"):
        x3 = conv(x2, channel * 4, kernel=4, stride=2, pad=1, pad_type='reflect')
        x3 = instance_norm(x3)
        x3 = tf.nn.relu(x3)
        layers.append(x3)
        print('Style encoder 3 shape is ', x3.shape)
        print()

    with tf.variable_scope("style_encoder_4"):
        x4 = conv(x3, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x4 = instance_norm(x4)
        x4 = tf.nn.relu(x4)
        layers.append(x4)
        print('Style encoder 4 shape is ', x4.shape)
        print()

    with tf.variable_scope("style_encoder_5"):
        x5 = conv(x4, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x5 = instance_norm(x5)
        x5 = tf.nn.relu(x5)
        layers.append(x5)
        print('Style encoder 5 shape is ', x5.shape)
        print()

    with tf.variable_scope("style_encoder_6"):
        x6 = conv(x5, channel * 8, kernel=4, stride=2, pad=1, pad_type='reflect')
        x6 = instance_norm(x6)
        x6 = tf.nn.relu(x6)
        layers.append(x6)
        print('Style encoder 6 shape is ', x6.shape)
        print()

    return x6, layers

##################################################################################
# Decoder
##################################################################################
def create_decoder(content, style, generator_outputs_channels, cnt_layers, sty_layers, a):
    channel = a.ngf * 16

    x = tf.concat([content, style], axis=3)
    print('Decoder input shape is ', x.shape)
    print()

    with tf.variable_scope("decoder_1"):
        x1 = up_sample(x, scale_factor=2)
        x1 = conv(x1, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect')
        x1 = instance_norm(x1)
        x1 = tf.nn.relu(x1)
        print('Decoder 1 shape is ', x1.shape)
        print()

    channel = channel // 2

    x1 = tf.concat([x1, cnt_layers[4], sty_layers[4]], axis=3)
    with tf.variable_scope("decoder_2"):
        x2 = up_sample(x1, scale_factor=2)
        x2 = conv(x2, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect')
        x2 = instance_norm(x2)
        x2 = tf.nn.relu(x2)
        print('Decoder 2 shape is ', x2.shape)
        print()

    channel = channel // 2

    x2 = tf.concat([x2, cnt_layers[3], sty_layers[3]], axis=3)
    with tf.variable_scope("decoder_3"):
        x3 = up_sample(x2, scale_factor=2)
        x3 = conv(x3, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect')
        x3 = instance_norm(x3)
        x3 = tf.nn.relu(x3)
        print('Decoder 3 shape is ', x3.shape)
        print()

    channel = channel // 2

    x3 = tf.concat([x3, cnt_layers[2], sty_layers[2]], axis=3)
    with tf.variable_scope("decoder_4"):
        x4 = up_sample(x3, scale_factor=2)
        x4 = conv(x4, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect')
        x4 = instance_norm(x4)
        x4 = tf.nn.relu(x4)
        print('Decoder 4 shape is ', x4.shape)
        print()

    channel = channel // 2

    x4 = tf.concat([x4, cnt_layers[1], sty_layers[1]], axis=3)
    with tf.variable_scope("decoder_5"):
        x5 = up_sample(x4, scale_factor=2)
        x5 = conv(x5, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect')
        x5 = instance_norm(x5)
        x5 = tf.nn.relu(x5)
        print('Decoder 5 shape is ', x5.shape)
        print()

    x5 = tf.concat([x5, cnt_layers[0], sty_layers[0]], axis=3)
    x6 = conv(x5, channels=generator_outputs_channels, kernel=7, stride=1, pad=3, pad_type='reflect')
    x6 = tf.tanh(x6)
    print('Decoder output shape is ', x6.shape)
    print()
    return x6


##################################################################################
# Discriminator
# Its a PatchGAN with outputs a patch of N*N dimension. N*N here is 30*30
# Each pixel in the N*N patch is actually telling whether the corresponding patch 
# in the input image is Real or Fake
##################################################################################

def create_discriminator(discrim_inputs, discrim_targets, args):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, args.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = args.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    # Adding Fully connected layer after our Encoder as we want to add classification loss
    output_flat = tf.layers.flatten(layers[-1])

    with tf.variable_scope('layer_fc_s'):
        styl_y = tf.layers.dense(output_flat, total_styles,
                                      kernel_initializer=tf.random_normal_initializer(0, 0.02))

    return layers[-1], styl_y

##################################################################################
# Build Model
# Run the Generator, then the discriminator two times for real and fake image respectively. 
# Two loss functions are used 1) GAN loss 2) L1 loss
# Then Generator and Discriminator are trained using the Adam Optimizer
# Then apply ExponentailMovingAverage while training the weights
# global_step then just keeps track of the number of batches seen so far.
##################################################################################

def create_model(src_font, targets, src_1stSpt, src_2ndSpt, src_3rdSpt, style_labels, char_labels, args):
    out_channels = int(targets.get_shape()[-1])
    with tf.name_scope("content_encoder_generator"):
        with tf.variable_scope("generator"):
            content_features, content_layers = create_content_enc(src_font, args)
    with tf.name_scope("style_encoder_generator"):
        with tf.variable_scope("generator"):
            style_features, style_layers = create_style_enc(src_1stSpt, src_2ndSpt, src_3rdSpt, args)
    with tf.name_scope("decoder_generator"):
        with tf.variable_scope("generator"):
            outputs = create_decoder(content_features, style_features, out_channels, content_layers, style_layers, args)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            predict_real, real_styl = create_discriminator(src_font, targets, args)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            predict_fake, _ = create_discriminator(src_font, outputs, args)

    # Loss Functions
    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predict_real), logits=predict_real))
        disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predict_fake), logits=predict_fake))
        
        disc_loss_real_styl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=style_labels, logits=real_styl))

        # Discriminator Final Loss
        discrim_loss =  disc_real_loss + disc_fake_loss + disc_loss_real_styl * args.classification_penalty

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predict_fake), logits=predict_fake))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * args.gan_weight + gen_loss_L1 * args.l1_weight

    # Training D and G using Adam
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([disc_real_loss, disc_fake_loss, disc_loss_real_styl, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        disc_real_loss=ema.average(disc_real_loss),
        disc_fake_loss=ema.average(disc_fake_loss),
        disc_loss_real_styl=ema.average(disc_loss_real_styl),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )