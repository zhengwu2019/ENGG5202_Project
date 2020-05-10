import os
import torch
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data.dataset import Dataset


def draw_hist(data_list):
    bins = np.linspace(0, 5, 100)
    for i, data in enumerate(data_list):
        plt.hist(data, bins, alpha=0.5, label=str(i))
    plt.legend(loc='upper right')
    plt.show()


def logits_analysis(pred_data):
    #pred_data: [N, 103]
    correct_mask = pred_data[:, 1] == 1
    num_correct = correct_mask.sum()
    acc = num_correct / pred_data.shape[0]
    print("Accuracy: {:.2f}%".format(acc*100))

    normalized_pred = F.softmax(torch.from_numpy(pred_data[:, -100:]))
    entropy = -normalized_pred * normalized_pred.log2()
    entropy = entropy.sum(dim=1)

    correct_entropy = entropy[correct_mask]
    incorrect_entropy = entropy[~correct_mask]
    draw_hist([correct_entropy, incorrect_entropy])


def get_pred(model_name, file_name, pkl_dir):
    file_name = file_name.replace(file_name.split('_')[0], model_name)
    file_path = os.path.join(pkl_dir, file_name)
    with open(file_path, 'rb') as f:
        pred_data = pkl.load(f)
        f.close()
    return pred_data

def get_metric_mask(pred_data, thres, metric='entropy'):
    # if pred_data.dtype == torch.float:
    #     pred_data = pred_data.cpu().numpy()
    if metric == "entropy":
        # 75.35, thres = [0.25, 0.25, 0.45, 0.2]
        normalized_pred = F.softmax(torch.from_numpy(pred_data[:, -100:]), dim=1)
        entropy = -normalized_pred * normalized_pred.log2()
        entropy = entropy.sum(dim=1)
        return entropy <= thres
    elif metric == 'weak_sum':
        # 75.73, thres = [0.08, 0.1, 0.25, 0.2],
        normalized_pred = F.softmax(torch.from_numpy(pred_data[:, -100:]), dim=1)
        max_pred_value, max_pred_index = normalized_pred[:, -100:].max(axis=1)
        sum_pred = normalized_pred[:, -100:].sum(axis=1)
        weak_sum_pred = sum_pred - max_pred_value
        return weak_sum_pred <= thres
    elif metric == 'mixed':
        normalized_pred = F.softmax(torch.from_numpy(pred_data[:, -100:]), dim=1)
        entropy = -normalized_pred * normalized_pred.log2()
        entropy = entropy.sum(dim=1)
        max_pred_value, max_pred_index = normalized_pred[:, -100:].max(axis=1)
        sum_pred = normalized_pred[:, -100:].sum(axis=1)
        weak_sum_pred = sum_pred - max_pred_value
        mask = (entropy < thres[0]) | (weak_sum_pred < thres[1])
        draw_hist([entropy[pred_data[:, 1]==1], weak_sum_pred[pred_data[:, 1]==1], entropy[pred_data[:, 1]!=1], weak_sum_pred[pred_data[:, 1]!=1]])
        return mask
    else:
        raise NotImplementedError


def assign_pred(pred_data, rest_mask, thres, model_name="resnet", final_model=False):
    metric_mask = get_metric_mask(pred_data, thres, metric='weak_sum')
    if not final_model:
        mask = metric_mask
    else:
        mask = rest_mask
    if rest_mask is not None:
        mask = mask & rest_mask
        rest_num = rest_mask.sum()
        rest_mask[mask] = False
    else:
        rest_num = pred_data.shape[0]
        rest_mask = ~mask

    correctness = pred_data[mask, 1] == 1
    num_correct = correctness.sum()
    num_samples = mask.sum()
    acc = 100 * correctness.sum() / mask.sum().float()
    print("\n{} process {}/{} samples after filtering, accuracy: {:.2f} ".format(model_name, mask.sum(), rest_num, acc))
    return mask, rest_mask, num_correct, num_samples


def analyze_train_pred(pkl_dir="./pred_pkl/"):
    model_names = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
    file_name = model_names[0] + "_val_5_pred.pkl"
    rest_mask = None
    total_correct = 0
    total_samples = 0
    masks = []
    all_pred_data = []
    thres = [0.35, 0.3, 0.45, 0.2]

    for i, model_name in enumerate(model_names):
        pred_data = get_pred(model_name, file_name, pkl_dir)
        all_pred_data.append(pred_data)
        mask, rest_mask, num_correct, num_samples = assign_pred(pred_data, rest_mask, thres[i], model_name, i==len(model_names) - 1)
        total_correct += num_correct
        total_samples += num_samples
        masks.append(mask)
    print("\ntotal_correct_num: {}, total_samples: {}, filtered accuracy: {:.2f}, total accuracy: {:.2f}".format(total_correct, total_samples,\
     total_correct*100 / total_samples.float(), total_correct*100 / float(pred_data.shape[0])))

    mask = pred_data[:, 1]
    for pred_data in all_pred_data:
        mask += pred_data[:, 1]
    res = mask > 0
    for i, mask in enumerate(masks):
        coorect_num = all_pred_data[i][mask][:, 1].sum()
        print(coorect_num, all_pred_data[i][:, 1].sum())

def analyze_test_pred(pkl_dir="./pred_pkl/"):
    model_names = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
    file_name = model_names[0] + "_val_-1_pred.pkl"
    rest_mask = None
    total_correct = 0
    total_samples = 0
    masks = []
    all_pred_data = []
    #thres = [0.25, 0.25, 0.45, 0.2]  # for entropy
    thres = [0.1, 0.1, 0.25, 1000]  # for weak_sum
    #thres = [[0.25, 0.08], [0.25, 0.1], [0.45, 0.25], [0.2, 0.2]]

    for i, model_name in enumerate(model_names):
        pred_data = get_pred(model_name, file_name, pkl_dir).cpu().numpy()
        all_pred_data.append(pred_data)
        mask, rest_mask, num_correct, num_samples = assign_pred(pred_data, rest_mask, thres[i], model_name, i==len(model_names)-1)
        total_correct += num_correct
        total_samples += num_samples
        masks.append(mask)

    print("\ntotal_correct_num: {}, total_samples: {}, filtered accuracy: {:.2f}, total accuracy: {}".format(total_correct, total_samples,\
     total_correct * 100 / total_samples.float(), total_correct * 100 / pred_data.shape[0]))

    num = 0
    mask = pred_data[:, 1]
    for pred_data in all_pred_data:
        mask += pred_data[:, 1]
    res = mask > 0
    #import ipdb; ipdb.set_trace()

def analyze_pred_res():
    model_names = ['resnet18', 'resnet34', 'resnet101', 'resnet152']

    # pred_thres = [0.55, 0.5, 0.5, -1]       # 74.00
    # ent_thres =  [0.25, 0.25, 0.45, 1000]   # 74.34
    # score_thres = [0.1, 0.1, 0.25, 1000]   # 74.60

    pred_thres = [0.7, 0.65, 0.6, -1]       # 74.78
    ent_thres =  [0.1, 0.1, 0.2, 1000]
    score_thres = [0.08, 0.1, 0.26, 1000]

    rest_mask = None
    total_correct_num = 0

    for i, model_name in enumerate(model_names):
        file_name = model_name + "_val_-1_pred.pkl"
        pred_data = get_pred(model_name, file_name, "./pred_pkl/").cpu().numpy()

        data_path = "./pred_pkl/" + model_name + "_assign_res_data.pkl"
        with open(data_path, 'rb') as f:
            assigner_res, assigner_out = pkl.load(f)
            f.close()
        pred_mask = assigner_out > pred_thres[i]
        pred_mask = torch.from_numpy(pred_mask)

        ent_mask = get_metric_mask(pred_data, ent_thres[i], metric='entropy')
        score_mask = get_metric_mask(pred_data, score_thres[i], metric='weak_sum')

        cat_mask = pred_mask | ent_mask | score_mask
        #cat_mask = pred_mask
        if rest_mask is None:
            rest_mask = ~cat_mask
        else:
            cat_mask[~rest_mask] = False
            pred_mask[~rest_mask] = False
            ent_mask[~rest_mask] = False
            score_mask[~rest_mask] = False
            rest_mask[cat_mask] =  False

        print(model_name)
        total_correct_num += pred_data[:, 1][cat_mask].sum()
        acc = pred_data[:, 1][cat_mask].sum() / cat_mask.sum().float()
        acc_0 = pred_data[:, 1][pred_mask].sum() / pred_mask.sum().float()
        acc_1 = pred_data[:, 1][ent_mask].sum() / ent_mask.sum().float()
        acc_2 = pred_data[:, 1][score_mask].sum() / score_mask.sum().float()


        print("total: {:.3f}, {:.0f}, {:.0f}".format(acc.cpu().numpy(), pred_data[:, 1][cat_mask].sum(), cat_mask.sum().cpu().numpy(),))
        print("pred:  {:.3f}, {:.0f}, {:.0f}".format(acc_0.cpu().numpy(), pred_data[:, 1][pred_mask].sum(), pred_mask.sum().cpu().numpy()))
        print("ent:   {:.3f}, {:.0f}, {:.0f}".format(acc_1.cpu().numpy(), pred_data[:, 1][ent_mask].sum(), ent_mask.sum().cpu().numpy()))
        print("score: {:.3f}, {:.0f}, {:.0f}".format(acc_2.cpu().numpy(),pred_data[:, 1][score_mask].sum(), score_mask.sum().cpu().numpy()))
        print("rest:  {}".format(rest_mask.sum()), '\n')


    print(total_correct_num)



        #import ipdb; ipdb.set_trace()





class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        super(SigmoidFocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor):

        per_entry_cross_ent = torch.clamp(prediction_tensor, min=0) - prediction_tensor * target_tensor.type_as(prediction_tensor)
        per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(prediction_tensor)))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities))
        modulating_factor = 1.0

        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)

        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha)

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)

        return focal_cross_entropy_loss

############################################# For Assginer Dataset #########################################################

def balance_assigner_train_dataset(assigner_data):
    data_seq_num = np.arange(assigner_data.shape[0])
    aligned_size = 3500
    max_label = assigner_data[:, 1].max().cpu().numpy()
    resamples_seq = data_seq_num.copy()
    for i in np.arange(0, max_label+1):
        mask_i = (assigner_data[:, 1] == i).cpu().numpy()
        num_i = mask_i.sum()
        resample_num = aligned_size - num_i
        resamples_seq_i = np.random.choice(data_seq_num[mask_i], size=resample_num, replace=True)
        resamples_seq = np.concatenate([resamples_seq, resamples_seq_i], axis=0)

    balanced_data = assigner_data[resamples_seq]

    return balanced_data


def generate_assigner_train_dataset(pkl_dir="./pred_pkl/", need_balance=True):
    # 0: 3403, 1: 109, 2: 208, 3: 577, 0: 703
    model_names = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
    for model_name in model_names[0:1]:
        file_name = model_name + "_val_5_pred.pkl"
        assigner_data = get_pred(model_name, file_name, pkl_dir)

        #discard: this part is for multi-class labeling
        # mask_0 = assigner_data[:, 1] == 1
        # unmask = ~mask_0   # unlabel mask
        # assigner_data[mask_0, 1] = 1
        # assigner_data[unmask, 1] = 0
        # for i, model_name in enumerate(model_names):
        #     pred_data = get_pred(model_name, file_name, pkl_dir)
        #     mask_i = pred_data[:, 1] == 1
        #     mask_i[~unmask] = False
        #     assigner_data[mask_i, 1] = i + 1

        # data_path = "./data/assigner_train_dataset_ml_wo_balanced.pkl"
        # if need_balance:
        #     assigner_data = balance_assigner_train_dataset(assigner_data)
        #     data_path = "./data/assigner_train_dataset_ml_balanced.pkl"

        assigner_data[:, -100:] = F.softmax(assigner_data[:, -100:], dim=1)
        data_path = "./data/" + model_name + "_assigner_train_dataset.pkl"
        with open(data_path, 'wb') as f:
            pkl.dump(assigner_data.cpu().numpy(), f)
            f.close()
    return

def generate_assigner_test_dataset(pkl_dir="./pred_pkl/"):
    # 0: 3403, 1: 109, 2: 208, 3: 577, -1: 703
    model_names = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
    for model_name in model_names[0:1]:
        file_name = model_name + "_val_-1_pred.pkl"
        assigner_data = get_pred(model_name, file_name, pkl_dir)

        # discard: this part is for multi-class labeling
        # mask_0 = assigner_data[:, 1] == 1
        # unmask = ~mask_0                  # unlabel mask
        # assigner_data[mask_0, 1] = 1      # mean the network can get correct prediction
        # assigner_data[unmask, 1] = 0      # mean the network can get incorrect prediction
        # for i, model_name in enumerate(model_names):
        #     pred_data = get_pred(model_name, file_name, pkl_dir)
        #     mask_i = pred_data[:, 1] == 1
        #     mask_i[~unmask] = False
        #     assigner_data[mask_i, 1] = i+1
        # data_path = "./data/assigner_test_dataset_ml.pkl"

        assigner_data[:, -100:] = F.softmax(assigner_data[:, -100:], dim=1)
        data_path = "./data/" + model_name + "_assigner_test_dataset_ml.pkl"
        with open(data_path, 'wb') as f:
            pkl.dump(assigner_data.cpu().numpy(), f)
            f.close()
    return

class AssginerData(Dataset):
    def __init__(self, train=True, model_name='resnet18'):
        self.train = train
        if train:
            root="./data/" + model_name + "_assigner_train_dataset.pkl"
            #root = './data/assigner_train_dataset_ml_balanced.pkl'
        else:
            root="./data/" + model_name + "_assigner_test_dataset.pkl"
            #root = './data/assigner_test_dataset_ml.pkl'

        with open(root, 'rb') as f:
            assigner_data = pkl.load(f)
            f.close()

        self.data = assigner_data[:, -100:]
        self.targets = assigner_data[:, 1]




    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)


############################################# END: For Assginer Dataset #########################################################


############################################# For Assginer Network  ############################################################

class AssginerNet(nn.Module):

    def __init__(self, num_classes=1):

        super(AssginerNet, self).__init__()
        self.linear1 = nn.Linear(100, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.linear2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.linear = nn.Linear(128, num_classes)


    def forward(self, x):

        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear(x)
        return x

############################################# END: For Assginer Network  ############################################################


#################################################### Assigner Trainer  ##############################################################
def assigner_trainer(train_mode=True, model_name='resnet152'):

    train_loader = torch.utils.data.DataLoader(
        AssginerData(train=True, model_name=model_name),
        batch_size=128, shuffle=True,
        num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        AssginerData(train=False, model_name=model_name),
        batch_size=128, shuffle=False,
        num_workers=1, pin_memory=True)

    checkpoint_path = "./checkpoints/" + model_name + "/" + model_name + "_cofidence_pred_checkpoint.pth"
    model = torch.nn.DataParallel(AssginerNet()).cuda()

    if train_mode:
        total_epoches = 200


        model.train()

        criterion = SigmoidFocalLoss().cuda()
        #criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.1)

        epoch_acc = 0
        for epoch in range(total_epoches):
            lr_scheduler.step()
            epoch_loss = 0

            for i, (input, target) in enumerate(train_loader):

                target_var = target.cuda().long()
                input_var = input.cuda()

                # compute output
                output = model(input_var).squeeze()
                loss = criterion(output, target_var)
                loss = loss.sum() / output.shape[0]

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.float()
                loss = loss.float()
                epoch_loss += loss

                pred_res = F.sigmoid(output).cpu() > 0.5
                #pred_res = output.max(dim=1)[1].cpu()

                acc = (pred_res == target).sum() / float(target.shape[0])
                epoch_acc = (epoch_acc * i  + acc ) / (i+1)

                if i == len(train_loader)-1:
                    print("epoch: {}, iteration: {}, loss: {}, accuracy: {}".format(epoch, i, epoch_loss, epoch_acc))

            if (epoch+1) == total_epoches:
                assigner_val(test_loader, model, model_name)

            if epoch == total_epoches-1:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    }, checkpoint_path)


    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        assigner_val(test_loader, model, model_name)



def assigner_val(test_loader, model, model_name):

    model.eval()
    assigner_res = np.zeros_like(test_loader.dataset.targets)
    assigner_out = np.zeros_like(test_loader.dataset.targets)

    with torch.no_grad():
        correct_num = 0
        for i, (input, target) in enumerate(test_loader):
            target_var = target.cuda().long()
            input_var = input.cuda()
            output = model(input_var)
            output = output.float()
            pred_res = F.sigmoid(output).cpu() > 0.5
            #pred_res = output.max(dim=1)[1].cpu()
            correct_num += (pred_res == target).sum()
            if i == len(test_loader) - 1:
                end_index = test_loader.dataset.data.shape[0]
            else:
                end_index = (i+1) * input.shape[0]
            assigner_res[i*test_loader.batch_size : end_index] = pred_res.squeeze()
            assigner_out[i*test_loader.batch_size : end_index] = F.sigmoid(output).cpu().squeeze()

        if model_name is not None:
            data_path = "./pred_pkl/" + model_name + "_assign_res_data.pkl"
            with open(data_path, 'wb') as f:
                pkl.dump([assigner_res, assigner_out], f)
                f.close()
                print("finish dumping assign results!")




if __name__ == '__main__':
    #main()
    # analyze_train_pred()
    # analyze_test_pred()
    # generate_assigner_train_dataset()
    # generate_assigner_test_dataset()
    # dataset = AssginerData(train=False)
    # import ipdb; ipdb.set_trace()
    # assigner_trainer()
    analyze_pred_res()
    analyze_test_pred()