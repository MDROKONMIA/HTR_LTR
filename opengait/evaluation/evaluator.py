import csv
import os
import pickle

import numpy as np
from utils import get_msg_mgr, mkdir

from .metric import cuda_dist, compute_ACC_mAP, evaluate_many


def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
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
def single_view_gallery_evaluation(feature_array, label_array, sequence_type_array, view_array,
                                   dataset_name, metric_function, dataset_base_path, msg_mgr=None):
    probe_sequences = {
        'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
        'OULP': {'NM': ['seq01']}
    }

    gallery_sequences = {
        'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
        'OULP': ['seq00']
    }

    accuracy_matrix = {}
    msg_mgr = get_msg_mgr()
    view_list = sorted(np.unique(view_array))
    top_rank = 1

    # --- Step 1: Preload all gallery valid frame counts ---
    gallery_frame_count = {}
    for label in np.unique(label_array):
        gallery_frame_count[label] = {}
        for gallery_seq in gallery_sequences[dataset_name]:
            gallery_frame_count[label][gallery_seq] = {}
            for gallery_view in view_list:
                path = os.path.join(dataset_base_path, str(label), gallery_seq, str(gallery_view))
                if os.path.exists(path):
                    try:
                        with open(os.path.join(path, f"{gallery_view}.pkl"), 'rb') as f:
                            data = np.array(pickle.load(f))
                        gallery_frame_count[label][gallery_seq][gallery_view] = len(data)
                    except Exception:
                        gallery_frame_count[label][gallery_seq][gallery_view] = -1
                else:
                    gallery_frame_count[label][gallery_seq][gallery_view] = -1

    sequence_type_array_copy = np.array(sequence_type_array)

    # --- Step 2: Evaluation Loop ---
    for gait_condition, probe_sequence_list in probe_sequences[dataset_name].items():
        accuracy_matrix[gait_condition] = np.full((len(view_list), len(view_list), top_rank), -1.0)

        for probe_view_index, probe_view in enumerate(view_list):
            probe_mask = np.isin(sequence_type_array, probe_sequence_list) & (view_array == probe_view)
            probe_features = feature_array[probe_mask]
            probe_labels = label_array[probe_mask]

            for gallery_view_index, gallery_view in enumerate(view_list):
                gallery_mask = np.isin(sequence_type_array, gallery_sequences[dataset_name]) & (view_array == gallery_view)
                gallery_features = feature_array[gallery_mask]
                gallery_labels = label_array[gallery_mask]
                gallery_sequences_selected = sequence_type_array_copy[gallery_mask]

                if len(probe_features) == 0 or len(gallery_features) == 0:
                    continue

                # Compute distance matrix and get sorted indices
                distance_matrix = cuda_dist(probe_features, gallery_features, metric_function)
                sorted_indices = distance_matrix.cpu().sort(1)[1].numpy()

                # --- Vectorized probe filtering ---
                # Create a 2D array: (num_probes, num_gallery_sequences)
                valid_matrix = np.zeros((len(probe_labels), len(gallery_sequences[dataset_name])), dtype=bool)
                for seq_idx, candidate_seq in enumerate(gallery_sequences[dataset_name]):
                    # Lookup frame counts for all probes
                    frames = np.array([gallery_frame_count[label].get(candidate_seq, {}).get(gallery_view, -1)
                                       for label in probe_labels])
                    valid_matrix[:, seq_idx] = frames >= 15

                # Any valid gallery sequence per probe
                valid_probe_mask = np.any(valid_matrix, axis=1)

                # Apply mask
                filtered_probe_labels = probe_labels[valid_probe_mask]
                filtered_sorted_indices = sorted_indices[valid_probe_mask]

                total_samples = len(filtered_probe_labels)
                if total_samples > 0:
                    correct_predictions = np.sum(
                        np.cumsum(
                            np.reshape(filtered_probe_labels, [-1, 1]) == gallery_labels[filtered_sorted_indices[:, 0:top_rank]], axis=1
                        ) > 0,
                        axis=0
                    )
                    accuracy_matrix[gait_condition][probe_view_index, gallery_view_index,:] = np.round(
                        correct_predictions * 100.0 / total_samples, 2
                    )

    # --- Step 3: Result Logging ---
    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    num_rank = top_rank
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_sequences[dataset_name].keys():
            sub_acc = de_diag(accuracy_matrix[type_][:,:,rank], each_angle=True)
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

    if dataset not in ('CASIA-B', 'OUMVLP'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric, dataset_path)



def evaluate_CCPG(data, dataset,dataset_path, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    seq_type = np.array(seq_type)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], ["U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank]) - 1.

    cmc_save, ap_save, minp = [], [], []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]
        gallery_seq_list = seq_type[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]
        probe_seq_list = seq_type[pseq_mask]

        msg_mgr.log_info(("gallery length", len(gallery_y), gallery_seq, "probe length", len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()

        cmc, ap, inp = evaluate_many(distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        minp.append(inp)

    msg_mgr.log_info(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        cmc_save[0] * 100, cmc_save[1] * 100, cmc_save[2] * 100, cmc_save[3] * 100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        ap_save[0] * 100, ap_save[1] * 100, ap_save[2] * 100, ap_save[3] * 100))

    msg_mgr.log_info(
        '===mINP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (minp[0] * 100, minp[1] * 100, minp[2] * 100, minp[3] * 100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 /
                    dist.shape[0], 2)
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