import os
import numpy as np
import argparse

experiment = os.path.join(os.getcwd(), 'experiment')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', '-n', type=int, default=10, help="number of epochs for training")
parser.add_argument('--batch_size', '-bs', type=int, default=32, help="batch size")
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="Adam optimizer learning rate")
parser.add_argument('--max_iter', type=int, default=2450, help="Save statistics and model checkpoint every N epochs")
parser.add_argument('--cnn', default='classifier', nargs='*', type=int, help="Specify the model architecture", required=True)
args = parser.parse_args()

# Define model
if args.cnn[0] == 0:
    model = 'classifier'
    if len(args.cnn) == 1:
        extra_path = '_default'
    else:
        extra_path = '_feature1_%d_feature2_%d_feature3_%d_hidden1_%d_hidden2_%d' % (args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5])
elif args.cnn[0] == 1:
    model = 'a'
    if len(args.cnn) == 1:
        extra_path = '_default'
    else:
        extra_path = '_feature1_%d_feature2_%d_feature3_%d_hidden1_%d_hidden2_%d' % (args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5])
elif args.cnn[0] == 2:
    model = 'b'
    if len(args.cnn) == 1:
        extra_path = '_default'
    else:
        extra_path = '_feature1_%d_feature2_%d_feature3_%d_hidden1_%d' % (args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4])
elif args.cnn[0] == 3:
    model = 'c'
    if len(args.cnn) == 1:
        extra_path = '_default'
    else:
        extra_path = '_feature1_%d_feature2_%d_feature3_%d_feature4_%d_hidden1_%d_hidden2_%d' % (args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5], args.cnn[6])
elif args.cnn[0] == 4:
    model = 'd'
    if len(args.cnn) == 1:
        extra_path = '_default'
    else:
        extra_path = '_feature1_%d_feature2_%d_feature3_%d_hidden_%d' % (args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4])
elif args.cnn[0] == 5:
    model = 'final'
    extra_path = 'final'
else:
    print("Unsupported") 
    assert(0)

def get_model_name(name, batch_size, learning_rate, epoch, batch_count):
    path = 'model_{0}_bs{1}_lr{2}_epoch{3}_iter{4}'.format(name, batch_size, learning_rate, epoch, batch_count)
    return os.path.join(experiment, path)


result_path = get_model_name(model, args.batch_size, args.learning_rate, args.num_epochs, args.max_iter) + extra_path

iteration = np.loadtxt("{}_iterations.csv".format(result_path))
train_acc = np.loadtxt("{}_train_accuracy.csv".format(result_path))
valid_acc = np.loadtxt("{}_valid_accuracy.csv".format(result_path))
train_loss = np.loadtxt("{}_train_loss.csv".format(result_path))
valid_loss = np.loadtxt("{}_valid_loss.csv".format(result_path))


for i in range(1, args.num_epochs+1):
    iterate = (args.max_iter+1)*i-1
    idx = np.where(iteration == iterate)
    print("Epoch %d, train accuracy %f validation accuracy %f | train loss %f validation loss %f" % (i, train_acc[idx], valid_acc[idx], train_loss[idx], valid_loss[idx]))
