import pandas as pd 
import numpy as np 
from scipy import linalg
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib
# matplotlib.use('Agg')  # headless run
import matplotlib.pyplot as plt


def analyse_semantic(probs, data_list, nseen):
    df = pd.read_csv(data_list)
    labels = np.array(df['label'].tolist())
    dlabels = np.array(df['dlabel']).tolist()
    labels = np.c_[labels, dlabels]
    preds = probs.argmax(axis=1)

    # real/fake detection
    detect = np.where(preds==0, 0, 1)
    grd = np.where(labels[:, 0]==0, 0, 1)
    acc_det = detect == grd
    print(f'Detection acc: {acc_det.mean()}, std: {acc_det.std()}')

    # attribution
    acc_attr = preds==labels[:, 0]
    print(f'Attribution acc: {acc_attr.mean()}, std {acc_attr.std()}')

    # seen
    ids = [i for i in range(len(labels)) if labels[i,1] in range(nseen)]
    p = preds[ids]
    g = labels[ids,0]
    acc_seen = p==g
    print(f'Attribution on {nseen} seen semantic: {acc_seen.mean()}, std: {acc_seen.std()}')

    # unseen
    nunseen = labels[:,1].max() + 1 - nseen  # any unseen semantics
    if nunseen > 0:
        ids = [i for i in range(len(labels)) if labels[i,1] not in range(nseen)]
        p = preds[ids]
        g = labels[ids,0]
        acc_unseen = p==g
        print(f'Attribution on {nunseen} unseen semantic: {acc_unseen.mean()}, std: {acc_unseen.std()}')
    else:
        print('No unseen semantic found.')
        acc_unseen = np.zeros(len(labels))

    msg = f'Seen Sem, Unseen Sem, Combined\n{acc_seen.mean():.4f}, {acc_unseen.mean():.4f}, {acc_attr.mean():.4f}'
    print(msg)
    return msg


def to_cuda(x):
    if type(x) is list:
        x = [item.cuda() for item in x]
    elif isinstance(x, dict):
        for key, val in x.items():
            if val is not None:
                x[key] = val.cuda()
    else:
        x = x.cuda()
    return x

def analyse(pred, labels, ndtrain):
    """
    used in test.py
    pred: predicted gan classes (N)
    labels: groundtruth gan and semantic (N,2)
    ndtrain: number of gan classes seen during training
    """
    # overall
    acc_all = np.mean(pred==labels[:,0])
    # seen sems
    ids = [i for i in range(len(labels)) if labels[i,1] in range(ndtrain)]
    p = pred[ids]
    g = labels[ids,0]
    acc_seen = np.mean(p==g)

    # unseen sems
    ids = [i for i in range(len(labels)) if labels[i,1] not in range(ndtrain)]
    p = pred[ids]
    g = labels[ids,0]
    acc_unseen = np.mean(p==g)
    msg = f'Seen Sem, Unseen Sem, Combined\n{acc_seen:.4f}, {acc_unseen:.4f}, {acc_all:.4f}'
    print(msg)
    return msg


def fdratio(feats, labels):
    """
    compute Fretche Distance ratio as specified at
    'Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints'
    https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Yu_Attributing_Fake_Images_ICCV_2019_supplemental.pdf
    Note: here we invert the score, so lower is better

    Input: feats    (N, D) features of N samples with featu dim D
           labels   (N)    labels of N samples
    Output: scalar higher is better
    """
    ulabels = np.unique(labels)
    ncats = len(ulabels)

    # intra-class FD
    fd_intra = 0
    intras = []
    rng = np.random.RandomState(7)
    for l in ulabels:
        feats_intra = feats[labels==l]
        ##  split into 2 groups
        n = len(feats_intra)
        ids = rng.permutation(n)
        intra1, intra2 = feats_intra[:(n//2)], feats_intra[(n//2):]
        ##  mean and variance
        meanvar1 = compute_mean_var(intra1)
        meanvar2 = compute_mean_var(intra2)
        ## FD score
        intra_ = compute_fd(meanvar1, meanvar2)
        fd_intra += intra_
        intras.append(intra_)
    fd_intra /= ncats

    # inter-class FD
    meanvars = [compute_mean_var(feats[labels==l]) for l in ulabels]
    fd_inter = 0
    inters = []
    for i in range(ncats-1):
        meanvar_i = meanvars[i]
        for j in range(i+1, ncats):
            meanvar_j = meanvars[j]
            inter_ = compute_fd(meanvar_i, meanvar_j)
            fd_inter += inter_
            inters.append(inter_)
    fd_inter /= (ncats * (ncats-1))
    return fd_intra / fd_inter, intras, inters   # here we swap nominator with denom


def compute_mean_var(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma 

def compute_fd(meanvar1, meanvar2):
    mu1, sigma1 = meanvar1
    mu2, sigma2 = meanvar2
    m = np.square(mu1-mu2).sum()
    s,_ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # pylint: disable=no-member
    dist = np.real(m + np.trace(sigma1 + sigma2 - 2*s))
    return dist 

def compute_purity(pred, labels):
    """
    compute purity of the predicted 'cluster'
    INPUT: pred  (N) predicted labels
           labels (N) groundtruth labels
    Output: scalar
    """
    conf = confusion_matrix(labels, pred)
    purity = 0
    for i in range(len(conf)):
        purity += conf[:, i].max()
    purity /= conf.sum()
    return purity 

def compute_nmi(pred, labels):
    """
    normalized mutual information
    """
    return normalized_mutual_info_score(labels, pred)

def plot_tsne(feats, labels, class_names=None, outpath=None):

    x_emb = TSNE(n_components=2).fit_transform(feats)

    fig, ax = plt.subplots()

    sc = ax.scatter(x_emb[:, 0], x_emb[:, 1], c=labels)
    if class_names is not None:
        ax.legend(handles=sc.legend_elements()[0], labels=class_names)
    # ax.set_title('GAN classes')
    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches = 'tight')
