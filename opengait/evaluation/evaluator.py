import os
import pickle

import numpy as np
from utils import get_msg_mgr, mkdir

from .metric import cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many


def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric, dataset_path):
    '''More details can be found: More details can be found in
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict


# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py

def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric, dataset_path):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']},
                      'OULP': {'NM': ['seq01']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],
                        'OULP': ['seq00']
                        }
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(view, probe_view) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(view, [gallery_view]) # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data, dataset, dataset_path, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'OULP'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric, dataset_path)
    else:
        return single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric, dataset_path)


def evaluate_Gait3D(data, dataset, dataset_path, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(
        open('./datasets/Gait3D/Gait3D.json', 'rb'))['PROBE_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    msg_mgr.log_info(results)
    return results

def evaluate_CCPG(data, dataset, dataset_path, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], [
        "U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.

    ap_save = []
    cmc_save = []
    minp = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        msg_mgr.log_info(
            ("gallery length", len(gallery_y), gallery_seq, "probe length", len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        minp.append(inp)

    # print(ap_save, cmc_save)

    msg_mgr.log_info(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        cmc_save[0]*100, cmc_save[1]*100, cmc_save[2]*100, cmc_save[3]*100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        ap_save[0]*100, ap_save[1]*100, ap_save[2]*100, ap_save[3]*100))

    msg_mgr.log_info(
        '===mINP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (minp[0]*100, minp[1]*100, minp[2]*100, minp[3]*100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)
    result_dict = {}
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Include identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[3, :, :, i])))
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: {}'.format(de_diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('UP: {}'.format(de_diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('DN: {}'.format(de_diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('BG: {}'.format(de_diag(acc[3, :, :, i], True)))
    return result_dict
