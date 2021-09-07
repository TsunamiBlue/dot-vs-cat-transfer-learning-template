import os
import matplotlib.pyplot as plt


def plotting():
    typeis = "pretrained"
    file_path = os.path.join("SSSS_p_loss.txt")
    with open(file_path,mode="r") as f:
        lines = f.readlines()
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        for idx, line in enumerate(lines):
            # print(line.split(','))
            item = line.split(',')
            # print(item)
            if idx>=20:
                break
            loss = float(item[1])
            acc = float(item[2])
            if idx <10:
                train_acc_list.append(acc)
                train_loss_list.append(loss)
            elif idx<20:
                val_acc_list.append(acc)
                val_loss_list.append(loss)
                # print(loss)
            else:
                break

        f.close()
    epochs = list(range(1,11))
    # fig, ax1 = plt.subplots()
    # ax1.plot(epochs, train_loss_list, 'r-')
    #
    # ax1.set_xlabel('epochs')
    # ax1.set_ylabel('training loss', color='r')
    #
    # # plt.savefig(f"{typeis}_train_loss")
    # # plt.show()
    #
    # fig2, ax1 = plt.subplots()
    # ax1.plot(epochs, train_acc_list, 'b-')
    #
    #
    # ax1.set_xlabel('epochs')
    # ax1.set_ylabel('training accuracy', color='b')
    #
    # # plt.savefig(f"{typeis}_train_acc")
    #
    fig3, ax1 = plt.subplots()
    ax1.plot(epochs, val_loss_list, 'r-')

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('validation loss', color='r')

    plt.savefig(f"{typeis}_val_loss")
    #
    # fig4, ax1 = plt.subplots()
    # ax1.plot(epochs, val_acc_list, 'b-')
    #
    # ax1.set_xlabel('epochs')
    # ax1.set_ylabel('validation accuracy', color='b')
    #
    # # plt.savefig(f"{typeis}_val_acc")



    plt.show()



    return 0






if __name__ == "__main__":
    plotting()