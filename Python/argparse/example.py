import argparse

def cli_main(flags):

    if len(flags.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpus

    train_records = flags.train_set_location
    validation_records = flags.validation_set_location

    train_df = DatasetGenerator(train_records, parse_function=parse_records, shuffle=True,
                                batch_size=flags.train_batchsize)
    validation_df = DatasetGenerator(validation_records, parse_function=parse_records, shuffle=True,
                                     batch_size=flags.validate_batchsize)

    reporter = Report()
    reporter.set_train_set(train_df)
    reporter.set_validation_set(validation_df)
    reporter.set_write_directory(flags.report_dir)
    reporter.set_ignore_list(["input", "depth"])
    with tf.compat.v1.Session() as sess:
        model = MNISTModel(sess, train_df, validation_df, learning_rate=flags.learning_rate, reporter=reporter)
        model.train(epochs=flags.epochs, save=flags.save, save_location=flags.model_save_dir)

    reporter.write_report(f"{flags.report_name_base}_train_{str(datetime.now())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='The number of epochs for training')

    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='The learning rate to use during training')

    parser.add_argument('--train_set_location', type=str,
                        default="./tf_records/train/mnist_train.tfrecords",
                        help='The location of the training set')

    parser.add_argument('--validation_set_location', type=str,
                        default="./tf_records/valid/mnist_valid.tfrecords",
                        help='The location of the validation set')

    parser.add_argument('--train_batchsize', type=int,
                        default=20,
                        help='The batch size to use for feeding training examples')

    parser.add_argument('--validate_batchsize', type=int,
                        default=20,
                        help='The batch size to use for feeding validation examples')

    parser.add_argument('--model_save_dir', type=str,
                        default="./model/mnist",
                        help='The directory to save the model in.')

    parser.add_argument('--report_name_base', type=str,
                        default="lenet-mnist",
                        help='The base name for the report')

    parser.add_argument('--save', type=bool,
                        default=True,
                        help='Whether or not to save the model')

    parser.add_argument('--report', type=bool,
                        default=True,
                        help='Whether to create a report.')

    parser.add_argument('--gpus', type=str,
                        default="",
                        help='Sets the GPU to use')

    parser.add_argument('--report_dir', type=str,
                        default='./reports/',
                        help='Where to save the reports.')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)