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
from sklearn.metrics import plot_confusion_matrix

# global variables
RANDOM_SEED = 1000
cuda = True if torch.cuda.is_available() else False
datapath = os.path.join(os.getcwd(), 'cleaned-dataset')
misclassified = os.path.join(os.getcwd(), 'misclassified')
class_examples = os.path.join(os.getcwd(), 'class_examples')
unknown = os.path.join(os.getcwd(), 'unknown')


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
    print("Total misclassified images: %d" % wrong)
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
            ax.label_outer()
        plt.subplots_adjust(wspace=-0.75, hspace=0.1)
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
    plt.subplots_adjust(hspace=0.1)


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
            img = inputs.squeeze().detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imsave(os.path.join(unknown, "unsure", "%d_pred_%d_label_%d.png" % (unsure, pred, label_class)), img)
            unsure_dict[label_class].append((unsure, pred))
            print("Unsure image %d, top three: %d (%1.3f), %d (%1.3f), %d (%1.3f)" % (unsure, top_three[0], top_three_prob[0], top_three[1], top_three_prob[1], top_three[2], top_three_prob[2]))
            unsure += 1
        else:
            if pred != label_class:
                img = inputs.squeeze().detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.imsave(os.path.join(unknown, "wrong", "%d_pred_%d_label_%d.png" % (wrong, pred, label_class)), img)
                wrong_dict[label_class].append((wrong, pred))
                # print("Wrong image %d, top three: %d (%1.3f), %d (%1.3f), %d (%1.3f)" % (unsure, top_three[0], top_three_prob[0], top_three[1], top_three_prob[1], top_three[2], top_three_prob[2]))
                wrong += 1
    print("Unsure images: %d. Misclassfied images: %d" % (unsure, wrong))
    return unsure_dict, wrong_dict

def get_confusion_matrix(net, loader):
    confusion_mat = np.zeros((24, 24), dtype=int)

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)
        # select index with maximim prediction score
        pred = outputs.max(1, keepdim=True)[1]
        prediction = int(pred.squeeze().detach().cpu())
        label_class = int(labels.squeeze().detach().cpu())
        actual_class = idx_to_class[label_class]
        predict_class = idx_to_class[prediction]

        confusion_mat[actual_class][predict_class] += 1

    row_sum = confusion_mat.sum(axis=1)
    confusion_mat = confusion_mat / row_sum[:, np.newaxis]

    plt.matshow(confusion_mat, cmap='Set3')
    plt.colorbar()
    tick_marks = np.arange(len(confusion_mat[0]))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy', type=bool, default=False)
    parser.add_argument('--analysis', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--unknown', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--confusion_mat', type=bool, default=False)
    args = parser.parse_args()

    report_test_accuracy = args.accuracy
    analyze_wrong_classifications = args.analysis
    visualize_activation = args.visualize
    test_with_unknown_label = args.unknown
    draw_confusion_matrix = args.confusion_mat

    # Define model
    model = FinalModel()
    model_path = os.path.join(os.getcwd(), 'traffic_sign_classifier')
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)

    if cuda:
        model = model.cuda()

    # Load test dataset
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

        count = 0
        for i, data in enumerate(test_loader, 0):
            if count == 1:
                inputs, labels = data
                print("Class label is: %d" % int(labels.squeeze().cpu()))
            count += 1
        print(count)

        if cuda:
            inputs = inputs.cuda()
        output = model(inputs)
        pred = output.max(1, keepdim=True)[1]
        print("Predicted class is: %d" % int(pred))

        fig = plt.imshow(inputs.squeeze().cpu().numpy().transpose((1,2,0)))
        plt.show()
        show_activation(6, 'conv1')
        show_activation(6, 'conv2')
        show_activation(6, 'conv3')
        plt.show()
    
    if test_with_unknown_label:
        threshold = args.threshold
        unsure_dict, wrong_dict = test_with_unknown_labels(threshold, model, test_loader)
        print("Visualizing images with low confidence")
        visualize_wrong_images(unsure_dict, idx_to_class, 1)
        print("Visualizing images with wrong prediction")
        visualize_wrong_images(wrong_dict, idx_to_class, 2)

    if draw_confusion_matrix:
        get_confusion_matrix(model, test_loader)

        # titles_options = [("Confusion matrix, without normalization", None),
                        #   ("Normalized confusion matrix", 'true')]
        # for title, normalize in titles_options:
            # disp = plot_confusion_matrix()