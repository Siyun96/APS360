import numpy as np
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
# import model
from final_model import FinalModel

# global variables
RANDOM_SEED = 1000
cuda = True if torch.cuda.is_available() else False
datapath = os.path.join(os.getcwd(), 'cleaned-dataset')
misclassified = os.path.join(os.getcwd(), 'misclassified')
class_examples = os.path.join(os.getcwd(), 'class_examples')
unknown = os.path.join(os.getcwd(), 'unknown')

def get_data_loader(batch_size=1):
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
    train_sampler = SubsetRandomSampler(train_indices[:10])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices[:10])
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def get_test_loader():
    # transform settings
    transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

    # set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # set up training set
    test_path = os.path.join(datapath, 'Test')
    test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    class_to_idx = test_set.class_to_idx
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    return test_loader, class_to_idx


def get_test_accuracy(net, loader):
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
    
    correct = correct / total_img

    return correct


def get_wrong_classifications(net, loader):
    wrong = 0
    label_dict = defaultdict(list)

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)
        # select index with maximim prediction score
        pred = outputs.max(1, keepdim=True)[1]
        if pred.eq(labels.view_as(pred)).sum().item() != 1:
            img = inputs.squeeze().detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            label_class = int(labels.squeeze().detach().cpu())
            prediction = int(pred.squeeze().detach().cpu())
            plt.imsave(os.path.join(misclassified, "%d_pred_%d_label_%d.png" % (wrong, prediction, label_class)), img)
            label_dict[label_class].append((wrong, prediction))
            wrong += 1
    print(wrong)
    return label_dict


def visualize_wrong_images(label_dict, idx_to_class, unknown_idx=0):
    if unknown_idx == 0:
        part_path = misclassified
    elif unknown_idx == 1:
        part_path = os.path.join(unknown, 'unsure')
    elif unknown_idx == 2:
        part_path = os.path.join(unknown, 'wrong')
    else:
        print("Unsupported")
        assert(0)
    for key in label_dict.keys():
        fig, axs = plt.subplots(2, len(label_dict[key])+1)
        axs[0, 0].imshow(mpimg.imread(os.path.join(class_examples, "%d.png"%idx_to_class[key])))
        for j in range(len(label_dict[key])):
            img_index, pred = label_dict[key][j]
            axs[0, j+1].imshow(mpimg.imread(os.path.join(part_path, "%d_pred_%d_label_%d.png" % (img_index, pred, key))))
            axs[1, j+1].imshow(mpimg.imread(os.path.join(class_examples, "%d.png" % (idx_to_class[pred]))))
        for ax in axs.flat:
            ax.set_axis_off()
        plt.show()


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def show_activation(num_rows, layer):
    act = activation[layer].squeeze()
    num_cols = act.size(0)//num_rows
    fig, axs = plt.subplots(num_rows, num_cols)
    for idx in range(act.size(0)):
        axs[idx//num_cols, idx%num_cols].imshow(act[idx].cpu())
    for ax in axs.flat:
        ax.set_axis_off()
        ax.label_outer()
    plt.subplots_adjust(wspace=-0.75, hspace=0.1)
    plt.show()


def test_with_unknown_labels(threshold, net, loader):
    unsure, wrong = 0, 0
    unsure_dict = defaultdict(list)
    wrong_dict = defaultdict(list)

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)

        # get top three class labels as well as their probabilities
        outputs = torch.nn.functional.softmax(outputs, dim=1).squeeze().detach().cpu()
        top_three_prob, top_three = torch.topk(outputs, 3)
        top_three = top_three.numpy()
        top_three_prob = top_three_prob.numpy()

        pred = top_three[0]
        pred_prob = top_three_prob[0]
        label_class = int(labels.squeeze().detach().cpu())

        if pred_prob < threshold:
            pred = 24
            img = inputs.squeeze().detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imsave(os.path.join(unknown, "unsure", "%d_pred_%d_label_%d.png" % (unsure, pred, label_class)), img)
            unsure_dict[label_class].append((unsure, pred))
            unsure += 1
        else:
            if pred != label_class:
                img = inputs.squeeze().detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.imsave(os.path.join(unknown, "wrong", "%d_pred_%d_label_%d.png" % (wrong, pred, label_class)), img)
                wrong_dict[label_class].append((wrong, pred))
                wrong += 1
    print("Unsure images: %d. Misclassfied images: %d" % (unsure, wrong))
    return unsure_dict, wrong_dict


if __name__ == "__main__":
    report_test_accuracy = False
    analyze_wrong_classifications = False
    visualize_activation = False
    test_with_unknown_label = True

    # Define model
    model = FinalModel()
    model_path = os.path.join(os.getcwd(), 'traffic_sign_classifier')
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)

    if cuda:
        model = model.cuda()
    test_loader, class_to_idx = get_test_loader()
    idx_to_class = {value:int(key) for key, value in class_to_idx.items()}
    idx_to_class[24] = 24

    if report_test_accuracy:
        test_accuracy = get_test_accuracy(model, test_loader)
        print("The model testing accuracy is: %f %%" % (test_accuracy*100))

    if analyze_wrong_classifications:
        label_dict = get_wrong_classifications(model, test_loader)
        visualize_wrong_images(label_dict, idx_to_class)

    if visualize_activation:
        model.conv1.register_forward_hook(get_activation('conv1'))
        model.conv2.register_forward_hook(get_activation('conv2'))
        model.conv3.register_forward_hook(get_activation('conv3'))

        it = iter(test_loader)
        data, _ = next(it)
        if cuda:
            data = data.cuda()
        output = model(data)

        show_activation(6, 'conv1')
        show_activation(6, 'conv2')
        show_activation(6, 'conv3')
    
    if test_with_unknown_label:
        threshold = 0.8
        unsure_dict, wrong_dict = test_with_unknown_labels(threshold, model, test_loader)
        print("Visualizing images with low confidence")
        visualize_wrong_images(unsure_dict, idx_to_class, 1)
        print("Visualizing images with wrong prediction")
        visualize_wrong_images(wrong_dict, idx_to_class, 2)