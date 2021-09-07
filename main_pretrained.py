import os
import numpy as np
import torch
import torchvision.transforms as T
from dataset import Picture
from torch.utils.data import dataloader,DataLoader
from model_pretrained import Pretrained_ResNet18
from torchvision import models
import torch.nn as nn
from tqdm import tqdm


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # required dataset for Q1-3
    test_pth = os.path.join("Cat-Dog-data", "cat-dog-test")
    train_pth = os.path.join("Cat-Dog-data", "cat-dog-train")
    original_train = Picture(train_pth)


    train_set, validation_set = torch.utils.data.random_split(original_train, [18000, 2000])
    test_set = Picture(test_pth)


    # Q4
    q4_transforms = T.Compose([T.ToPILImage(),
                               T.Resize(224),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    original_train_4 = Picture(train_pth, q4_transforms)
    train_set_4, validation_set_4 = torch.utils.data.random_split(original_train_4, [18000, 2000])
    test_set_4 = Picture(test_pth, q4_transforms)


    # Q5
    q5_transforms = T.Compose([T.ToPILImage(),
                               T.Resize(224),
                               T.RandomHorizontalFlip(0.5),
                               T.RandomCrop(size=(224,224), padding=4),

                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    original_train_5 = Picture(train_pth, q5_transforms)
    train_set_5, validation_set_5 = torch.utils.data.random_split(original_train_5, [18000, 2000])




    # hyperparams
    epochs = 10
    lr = 0.001
    batch_size = 200

    train_loader = dataloader.DataLoader(train_set_5, batch_size=batch_size, num_workers=4, drop_last=False,
                                         shuffle=True)
    validate_loader = dataloader.DataLoader(validation_set_5,batch_size=batch_size, num_workers=4, drop_last=False,
                                         shuffle=False)
    test_loader = dataloader.DataLoader(test_set_4,batch_size=2, num_workers=4, drop_last=True,
                                         shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18 = models.resnet18(pretrained=True)
    fc_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(fc_features, 2)
    print(f"Current device: {device}")
    network = resnet18
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()


    training_loss_list = []
    validation_loss_list = []


    # training
    with tqdm(range(1,epochs+1),ncols=100,ascii=True) as tq:
        for epoch in tq:

            num_batch = 0

            # lr_cos = lambda n: 0.5 * (1 + np.cos(n / epochs * np.pi)) * lr
            # current_lr = lr_cos(epoch)
            # for group in optimizer.param_groups:
            #     group['lr'] = current_lr
            network.train()
            epoch_loss = 0.0
            total = 0
            correct = 0
            for idx,(data,name,label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                forward_result = network(data)

                # print(forward_result.size())
                # print(forward_result,label)
                loss = criterion(forward_result,label)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()
                num_batch+=1

                # find train acc
                _, predict = torch.max(forward_result.data, 1)
                total += label.size(0)
                correct += predict.eq(label.data).cpu().sum()

                tq.set_description(
                    f"Training... Current Epoch {epoch} Current Batch: {idx*data.shape[0]}/{len(train_loader.dataset)} Loss:{loss.item()} train:{100. * correct/total}"
                )
            training_loss_list.append((epoch,epoch_loss/ (18000/batch_size),100. * correct/total))
            # with open("Q2logging.txt", mode="a+") as f:
            #     f.write(f"{epoch,epoch_loss/ (18000/batch_size),100. * correct/total}")

            network.eval()
            validate_loss = 0.0
            total = 0
            correct = 0
            with torch.no_grad():
                for batch_idx, (img, name, label) in enumerate(validate_loader):
                    img = img.to(device)
                    label = label.to(device)

                    validate_output = network(img)
                    v_loss = criterion(validate_output, label)
                    validate_loss += float(v_loss.item())
                    _, predict = torch.max(validate_output.data, 1)
                    total += label.size(0)
                    correct += predict.eq(label.data).cpu().sum()



                print(f"Validation epoch {epoch}  loss: {validate_loss / (2000 / batch_size)}  Accuracy: {100. * correct / total}")
                validation_loss_list.append((epoch,validate_loss / (2000 / batch_size),100. * correct/total))
                # with open("Q2logging.txt", mode="a+") as f:
                #     f.write(f"{epoch, validate_loss / (2000 / batch_size), 100. * correct / total}")

        print('Training is finished')

        # evaluate with test set
        network.eval()
        # test_loader = DataLoader(test_set_4, batch_size=batch_size, num_workers=2, drop_last=True, shuffle=True)
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (img, _, label) in enumerate(test_loader):
                img = img.to(device)
                label = label.to(device)
                test_output = network(img)
                loss_t = criterion(test_output, label)
                test_loss += float(loss_t.item())
                _, predicted = torch.max(test_output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            # with open("Q2.4_logging.txt", mode="a+") as file:
            #     file.write('Test After : loss: %.6f | Accuracy: %.3f%% \n' %
            #                    (running_loss_t / (4000/batch_size), 100. * correct / total))
            test_loss = test_loss / (4000 / batch_size)
            test_acc = 100. * correct / total
            print(f"Final Test: loss: {test_loss}  accuracy: {test_acc}")


    with open("Q2log_2.txt",mode="a+") as f:
        # f.writelines(training_loss_list)
        # f.writelines(validation_loss_list)
        for item in training_loss_list:
            f.write(f"{int(item[0])}, {float(item[1])}, {float(item[2])}  \n")
        for item in validation_loss_list:
            f.write(f"{int(item[0])}, {float(item[1])}, {float(item[2])}  \n")
        f.write(f"Final Result for test loss and acc|{test_loss}|{test_acc}")
        f.close()




if __name__ == "__main__":
    main()
