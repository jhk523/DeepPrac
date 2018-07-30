# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd


class Train(object):
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()

    def placeholders(self):
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                       IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def build_train_validation_graph(self):
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)

    def train(self):
        # For the first step, we are loading all training images and validation images into the
        # memory
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        vali_data, vali_labels = read_validation_data()

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print('Restored from checkpoint...')
        else:
            sess.run(init)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print('Start training...')
        print('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                                     FLAGS.train_batch_size)

            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                                                      vali_labels,
                                                                                      FLAGS.validation_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                                                                         top1_error=self.vali_top1_error,
                                                                                         vali_data=vali_data,
                                                                                         vali_labels=vali_labels,
                                                                                         session=sess,
                                                                                         batch_data=train_batch_data,
                                                                                         batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                                 self.vali_top1_error,
                                                                                 self.vali_loss],
                                                                                {
                                                                                    self.image_placeholder: train_batch_data,
                                                                                    self.label_placeholder: train_batch_labels,
                                                                                    self.vali_image_placeholder: validation_batch_data,
                                                                                    self.vali_label_placeholder: validation_batch_labels,
                                                                                    self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)

            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                  self.full_loss, self.train_top1_error],
                                                                 {self.image_placeholder: train_batch_data,
                                                                  self.label_placeholder: train_batch_labels,
                                                                  self.vali_image_placeholder: validation_batch_data,
                                                                  self.vali_label_placeholder: validation_batch_labels,
                                                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time

            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                              sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('Validation loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step': step_list, 'train_error': train_error_list,
                                        'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')

    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print('%i test batches in total...' % num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i batches finished!' % step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                              feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array

    ## Helper functions
    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def top_k_error(self, predictions, labels, k):
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch

    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset + FLAGS.train_batch_size]

        return batch_data, batch_label

    def train_operation(self, global_step, total_loss, top1_error):
        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        # This ema object help calculate the moving average of validation loss and error
        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))

        return val_op

    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                         self.vali_image_placeholder: vali_data_subset[offset:offset + FLAGS.validation_batch_size,
                                                      ...],
                         self.vali_label_placeholder: vali_labels_subset[offset:offset + FLAGS.validation_batch_size],
                         self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)


maybe_download_and_extract()
# Initialize the Train object
train = Train()
# Start the training session
train.train()




