import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        #self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(x.device)
        #if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss



# 自定义centerloss损失函数
class CenterLoss2(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2) -> None:
        super().__init__()
        self.cls_num = num_classes
        self.feat_num = feat_dim
        self.center = nn.Parameter(torch.randn(self.cls_num, self.feat_num)) # 中心点随机产生

    def forward(self, x, labels):
        center_exp = self.center.index_select(dim=0, index=labels.long()) # [N, 2]
        count = torch.histc(labels.float(), bins=self.cls_num, min=0, max=self.cls_num-1) # [10]
        count_exp = count.index_select(dim=0, index=labels.long())+1 # [N]
        # loss = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(x - center_exp,2), dim=1)), count_exp)) # 求损失, 原公式
        loss = torch.sum(torch.div(torch.sum(torch.pow(x - center_exp,2), dim=1), 2*count_exp)) # 求损失，略不同
        return loss 
    
