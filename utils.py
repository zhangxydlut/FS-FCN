import os
import torch
from pylab import plt


def check_dir(checkpoint_dir):#create a dir if dir not exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(os.path.join(checkpoint_dir,'model'))
        os.makedirs(os.path.join(checkpoint_dir,'pred_img'))


def loss_calc_v1(pred, label, gpu):

    label = label.long()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda(gpu)

    return criterion(pred, label)


def plot_loss(checkpoint_dir,loss_list,save_pred_every):
    x=range(0,len(loss_list)*save_pred_every,save_pred_every)
    y=loss_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Train loss')
    plt.xticks(range(0,len(loss_list)*save_pred_every+3,(len(loss_list)*save_pred_every+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'loss_fig.pdf'))
    plt.close()


def plot_iou(checkpoint_dir,iou_list):
    x=range(0,len(iou_list))
    y=iou_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='red',marker='o',label='IOU')
    plt.xticks(range(0,len(iou_list)+3,(len(iou_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'iou_fig.pdf'))
    plt.close()


def get_iou_v1(query_mask,pred_label,mode='foreground'):#pytorch 1.0 version
    if mode=='background':
        query_mask=1-query_mask
        pred_label=1-pred_label
    num_img=query_mask.shape[0]#batch size
    num_predict_list,inter_list,union_list,iou_list=[],[],[],[]
    for i in range(num_img):
        num_predict=torch.sum((pred_label[i]>0).float()).item()
        combination = (query_mask[i] + pred_label[i]).float()
        inter = torch.sum((combination == 2).float()).item()
        union = torch.sum((combination ==1).float()).item()+torch.sum((combination ==2).float()).item()
        if union!=0:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(inter/union)
            num_predict_list.append(num_predict)
        else:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(0)
            num_predict_list.append(num_predict)
    return inter_list,union_list,iou_list,num_predict_list