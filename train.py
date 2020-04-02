import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for gradient descent
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # for plotting

# import model
from classifier import Classifier

# global variables
RANDOM_SEED = 1000
cuda = True if torch.cuda.is_available() else False
datapath = os.path.join(os.getcwd(), 'cleaned-dataset')
experiment = os.path.join(os.getcwd(), 'experiment')


# load datasets
def get_data_loader(batch_size):
    # transform settings
    transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

    # set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # split training dataset into training (80%) and validation (20%)
    training_path = os.path.join(datapath, 'Train')
    train_set = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    indices_train = list(range(len(train_set)))
    np.random.shuffle(indices_train)
    split = int(len(indices_train) * 0.8)

    train_indices, val_indices = indices_train[:split], indices_train[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def get_model_name(name, batch_size, learning_rate, epoch, batch_count):
    path = 'model_{0}_bs{1}_lr{2}_epoch{3}_iter{4}'.format(name, batch_size, learning_rate, epoch, batch_count)
    return os.path.join(experiment, path)


def evaluate(net, loader, criterion):
    total_loss = 0
    correct = 0
    total_img = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)
        # select index with maximim prediction score
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total_img += inputs.shape[0]
        # calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    
    correct = correct / total_img
    loss = total_loss / (i+1)

    return correct, loss


def train_net(net, batch_size, learning_rate, num_epochs, sample_interval):
    # Fix random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Get batches of the datasets
    train_loader, val_loader = get_data_loader(batch_size)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Set up lists to store training/validation accuracy/loss
    iterations, train_acc, train_loss, val_acc, val_loss = [], [], [], [], []

    # Training loop
    start_time = time.time()
    total_batch_counter = 0
    for epoch in range(num_epochs):
        training_loss_per_epoch = 0
        train_batch = 0
        train_correct = 0
        total_img = 0
        for inputs, labels in iter(train_loader):
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            # Run the network
            outputs = net(inputs)

            # Compute loss and update gradients
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate loss
            training_loss_per_epoch += loss.item()

            # Update counts
            train_batch += 1
            total_batch_counter += 1

            # Calculate the statistics every N batches
            if train_batch % sample_interval == 0 or train_batch == len(train_loader) - 1:
                ## Training
                # select index with maximim prediction score
                pred = outputs.max(1, keepdim=True)[1]
                train_correct += pred.eq(labels.view_as(pred)).sum().item()
                total_img += inputs.shape[0]
                train_acc.append(train_correct / total_img)
                train_loss.append(training_loss_per_epoch / train_batch)

                ## Validation
                temp_acc, temp_loss = evaluate(net, val_loader, criterion)
                val_acc.append(temp_acc)
                val_loss.append(temp_loss)

                ## Count total batch number
                iterations.append(total_batch_counter)

                ## Save model (checkpoint)
                model_path = get_model_name(net.name, batch_size, learning_rate, epoch+1, train_batch)
                torch.save(net.state_dict(), model_path)
        
        # Write the training/validation accuracy/loss into CSV file for plotting later
        np.savetxt("{}_train_accuracy.csv".format(model_path), train_acc)
        np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
        np.savetxt("{}_valid_accuracy.csv".format(model_path), val_acc)
        np.savetxt("{}_valid_loss.csv".format(model_path), val_loss)
        np.savetxt("{}_iterations.csv".format(model_path), iterations)

    # Training complete
    print('Finished Training!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Total time elapse: {:.2f} seconds'.format(elapsed_time))

    return train_batch


def plot_training_curve(model_path):
    # Get batch count
    iteration = np.loadtxt("{}_iterations.csv".format(model_path))

    # Plot accuracy
    train_acc = np.loadtxt("{}_train_accuracy.csv".format(model_path))
    plt.plot(iteration, train_acc, label='Train')
    val_acc = np.loadtxt("{}_valid_accuracy.csv".format(model_path))
    plt.plot(iteration, val_acc, label='Validation')
    plt.title("Training vs. Validation Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("%s_accuracy.png" % model_path)

    plt.clf()
    # Plot loss
    train_loss = np.loadtxt("{}_train_loss.csv".format(model_path))
    plt.plot(iteration, train_loss, label='Train')
    val_loss = np.loadtxt("{}_valid_loss.csv".format(model_path))
    plt.plot(iteration, val_loss, label='Validation')
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("%s_loss.png" % model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', '-n', type=int, default=20, help="number of epochs for training")
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help="batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="Adam optimizer learning rate")
    parser.add_argument('--sample_interval', type=int, default=1000, help="Save statistics and model checkpoint every N epochs")
    parser.add_argument('--cnn', default='classifier', nargs='*', help="Specify the model architecture", required=True)
    args = parser.parse_args()

    # Define model
    if args.cnn[0] == 'classifier':
        if len(args.cnn) == 1:
            model = Classifier()
        else:
            model = Classifier(args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5])
    elif args.cnn[0] == 'a':
        if len(args.cnn) == 1:
            model = ModelA()
        else:
            model = ModelA(args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5])
    elif args.cnn[0] == 'b':
        if len(args.cnn) == 1:
            model = ModelB()
        else:
            model = ModelB(args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4])
    elif args.cnn[0] == 'c':
        if len(args.cnn) == 1:
            model = ModelC()
        else:
            model = ModelC(args.cnn[1], args.cnn[2], args.cnn[3], args.cnn[4], args.cnn[5], args.cnn[6])
    else:
        print("Unsupported") 
        assert(0)

    if cuda:
        model = model.cuda()
    
    # Train model
    max_iter = train_net(model, args.batch_size, args.learning_rate, args.num_epochs, args.sample_interval)

    # Plot training curves
    model_path = get_model_name(model.name, args.batch_size, args.learning_rate, args.num_epochs, max_iter)
    plot_training_curve(model_path)

