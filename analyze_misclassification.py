import numpy as np
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# import model
from classifier import Classifier

# global variables
RANDOM_SEED = 1000
cuda = True if torch.cuda.is_available() else False
datapath = os.path.join(os.getcwd(), 'cleaned-dataset')
experiment = os.path.join(os.getcwd(), 'experiment_0401')
misclassified = os.path.join(os.getcwd(), 'misclassified')

def get_test_loader():
    # transform settings
    transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

    # set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # set up training set
    test_path = os.path.join(datapath, 'Test')
    test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    return test_loader


def get_wrong_classifications(net, loader):
    wrong = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)
        # select index with maximim prediction score
        pred = outputs.max(1, keepdim=True)[1]
        if pred.eq(labels.view_as(pred)).sum().item() != 1:
            print(outputs)
            print(pred)
            print(labels)
            img = inputs.squeeze().detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imsave(os.path.join(misclassified, "%d_pred_%d_label_%d.png" % (wrong, pred.squeeze().detach().cpu(), labels.squeeze().detach().cpu())), img)
            wrong += 1


def visualize_kernels():
    pass

def get_model_name(name, batch_size, learning_rate, epoch, batch_count):
    path = 'model_{0}_bs{1}_lr{2}_epoch{3}_iter{4}'.format(name, batch_size, learning_rate, epoch, batch_count)
    return os.path.join(experiment, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', '-n', type=int, default=10, help="number of epochs for training")
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help="batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="Adam optimizer learning rate")
    parser.add_argument('--max_iter', type=int, default=2450, help="maximum number of iterations per epoch")
    args = parser.parse_args()

    # Define model
    model = Classifier()
    model_path = get_model_name(model.name, args.batch_size, args.learning_rate, args.num_epochs, args.max_iter) + "_default"
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)

    if cuda:
        model = model.cuda()

    test_loader = get_test_loader()

    get_wrong_classifications(model, test_loader)


