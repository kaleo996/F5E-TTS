import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    pred_k = (correct_k.mul_(100.0 / batch_size)).item()
    return torch.tensor(pred_k)

class SoftmaxLoss(nn.Module):
	def __init__(self, nOut, nClasses):
	    super(SoftmaxLoss, self).__init__()

	    self.test_normalize = True
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(nOut,nClasses)

	    print('Initialised Softmax Loss')

	def forward(self, x):

		x 		= self.fc(x)
#		nloss   = self.criterion(x, label)
#		prec1 = accuracy(x.detach().cpu(), label.detach().cpu())

		return x

class StatsPool(nn.Module):
    def __init__(self):
        super(StatsPool, self).__init__()
    def forward(self, x):# x: [B, fea, T]
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out

class AngleprotoLoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleprotoLoss, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss       = self.criterion(cos_sim_matrix, label)
        prec1    = accuracy(cos_sim_matrix.detach().cpu(),
                label.detach().cpu(), topk=(1,))

        return nloss, prec1    
    
class ProtoLoss(nn.Module):

    def __init__(self):
        super(ProtoLoss, self).__init__()

        self.test_normalize = False

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output      = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss       = self.criterion(output, label)
        prec1, _    = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))

        return nloss, prec1    
    
class SoftmaxprotoLoss(nn.Module):

    def __init__(self, in_features, out_features, nPerSpeaker=2):
        super(SoftmaxprotoLoss, self).__init__()

        self.test_normalize = True
        self.nPerSpeaker=nPerSpeaker
        self.softmax = SoftmaxLoss(in_features, out_features)
        self.angleproto = AngleprotoLoss()

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):
        x = x.reshape(-1, self.nPerSpeaker, x.size()[-1]).squeeze(1)
        #x = x.reshape(self.nPerSpeaker, -1, x.size()[-1]).transpose(1,0).squeeze(1)
        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.reshape(-1,x.size()[-1]),
                label)#label.repeat_interleave(2)

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1    

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets

    modified from : https://github.com/adambielski/siamese-triplet
    see more: https://omoindrot.github.io/triplet-loss#why-not-just-use-softmax
              https://gombru.github.io/2019/04/03/ranking_loss/
    """
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, input, label):
        embeddings = F.normalize(input)
        triplets = self.triplet_selector.get_triplets(embeddings, label)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        #ap_distances = 1-torch.abs(torch.sum(embeddings[triplets[:, 0]] * embeddings[triplets[:, 1]], dim=-1))  # .pow(.5)
        #an_distances = 1-torch.abs(torch.sum(embeddings[triplets[:, 0]] * embeddings[triplets[:, 2]], dim=-1))  # .pow(.5)
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class ArcMarginProductNonLinearSquarshing(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=0.50, s=30.0, easy_margin=False):
        super(ArcMarginProductNonLinearSquarshing, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # non-linear squashing: function vectors of small magnitude are shrunk to
        # almost zeros while vectors of big magnitude are normalized to the length
        # slightly below 1.
        input_norml2 = torch.pow(torch.norm(input, dim=1), 2)
        mul_s = (input_norml2 / (input_norml2 + 1.0)).unsqueeze(1)
        cosine = F.linear(mul_s * F.normalize(input), F.normalize(self.weight))

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.get_device())
        # one_hot = torch.zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        loss = F.cross_entropy(output, label)
        acc = phi.max(-1)[1].eq(label).sum().item() / len(
                label) * 100

        return loss, acc



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=0.50, s=30.0, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.get_device())
        # one_hot = torch.zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        loss = F.cross_entropy(output, label)
        acc = phi.max(-1)[1].eq(label).sum().item() / len(
                label) * 100

        return loss, acc

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """
    def __init__(self, in_features, out_features, m=0.40, s=30.0):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        loss = F.cross_entropy(output, label)
        acc = phi.max(-1)[1].eq(label).sum().item() / len(
            label) * 100
        return loss, acc


class MCDropout(nn.Module):
    def __init__(self, rate, dropout_on_infer):
        super().__init__()
        self.rate = rate
        self.dropout_on_infer = dropout_on_infer

    def forward(self, x):
        if self.dropout_on_infer:
            return F.dropout(x, self.rate)
        else:
            return F.dropout(x, self.rate, training=self.training)


class AMLinear(nn.Module):
    def __init__(self, in_features, n_cls, m, s=30):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, n_cls))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = n_cls
        self.s = s

    def forward(self, x, labels):
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = torch.mm(x, ww)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m
        # labels_one_hot = torch.zeros(len(labels), self.n_cls, device=device).scatter_(1, labels.unsqueeze(1), 1.)
        labels_one_hot = torch.zeros(len(labels), self.n_cls, device=labels.get_device()).scatter_(1, labels.unsqueeze(1), 1.)
        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        loss = F.cross_entropy(adjust_theta, labels)
        acc = cos_theta.max(-1)[1].eq(labels).sum().item() / len(
                labels) * 100        
        return loss, acc


def cnn_bn_relu(indim, outdim, kernel_size, stride=1, dilation=1):
    return nn.Sequential(
            nn.Conv1d(indim, outdim, kernel_size, stride=stride, dilation=dilation),
            nn.BatchNorm1d(outdim),
            torch.nn.ReLU(),
        )


def comp_memory_consumption(n_batch, n_length, n_hiddens, no_grad, n_byte=4,
                            backprop=False, batch_norm=True, with_activation=True, overhead=0.6):
    if no_grad:
        backprop, batch_norm = False, False
        n_hidden_max = max(n_hiddens)
        previous_max = n_hiddens[np.argmax(n_hiddens)-1]
        n_hiddens_tot = n_hidden_max + previous_max
    else:
        n_hiddens_tot = sum(n_hiddens)
    mem_allocated = n_batch * n_length * n_hiddens_tot \
                    * n_byte * (1 + backprop + with_activation + batch_norm) / 1e9
    print(f'predictive memory is {mem_allocated+overhead}GB')
    return mem_allocated+overhead



