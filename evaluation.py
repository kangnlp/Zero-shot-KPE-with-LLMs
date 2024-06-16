import os
import re
import argparse
import nltk
import json
import logging


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1

def print_PRF(P, R, F1, N):
    logging.info("\nN=" + str(N))
    logging.info("P=" + str(P))
    logging.info("R=" + str(R))
    logging.info("F1=" + str(F1) + "\n")
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Directory path of pred files")
    args = parser.parse_args()

    preds_dir_path = args.path
    log_file_path = os.path.join(preds_dir_path, 'experiment_results')

    dataset_list = ['Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin'] # 'Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin'

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file_path, 'w'))

    porter = nltk.PorterStemmer()

    files = os.listdir(preds_dir_path)    

    f_5_scores = []
    f_10_scores = []
    f_15_scores = []

    for dataset_name in dataset_list:
        logging.info(f"Dataset Name: {dataset_name}")
        pred_file_path = os.path.join(preds_dir_path, f"{dataset_name}_result.json")

        with open(pred_file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines()
            json_list = [json.loads(line.strip()) for line in lines]

        preds = [j_data['final_pred_keyphrase'] for j_data in json_list]
        labels = [j_data['label'] for j_data in json_list]


        #_, labels, _ = data.get_processed_data(data_name) 

        if len(preds) != len(labels):
            raise ValueError("The lengths of the preds and labels are not equal.")
        

        num_c_5 = num_c_10 = num_c_15 = 0
        num_e_5 = num_e_10 = num_e_15 = 0
        num_s = 0

        for pred_list, label_list in zip(preds, labels):
            
            pred_list = [ p.replace('-'," ") for p in pred_list ]
            pred_list = [ p.replace('\n',"") for p in pred_list ]
            pred_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in pred_list ]
            pred_list = [ " ".join(pred.split()) for pred in pred_list ]
            pred_list = [ p.lower().strip() for p in pred_list ]

            label_list = [ l.replace('-'," ") for l in label_list ]
            label_list = [ l.replace('\n',"") for l in label_list ]
            label_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in label_list ]
            label_list = [ " ".join(l.split()) for l in label_list ]
            label_list = [ l.lower().strip() for l in label_list ]

            pred_set = []
            for pred in pred_list:
                if pred in pred_set or pred =='':
                    continue
                else:
                    pred_set.append(pred)

            pred_set_list = pred_set[:15]

            pred_s_list = []
            for p in pred_set_list:
                tokens = p.split()
                pred_s_list.append(' '.join(porter.stem(t) for t in tokens))

            label_s_list = []
            for l in label_list:
                tokens = l.split()
                label_s_list.append(' '.join(porter.stem(t) for t in tokens))

            j = 0
            for pred, pred_s in zip(pred_set_list, pred_s_list):
                if pred_s in label_s_list or pred in label_list:
                    if (j < 5):
                        num_c_5 += 1
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 10 and j >= 5):
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 15 and j >= 10):
                        num_c_15 += 1
                j += 1

            if (len(pred_list[0:5]) == 5):
                num_e_5 += 5
            else:
                num_e_5 += len(pred_list[0:5])

            if (len(pred_list[0:10]) == 10):
                num_e_10 += 10
            else:
                num_e_10 += len(pred_list[0:10])

            if (len(pred_list[0:15]) == 15):
                num_e_15 += 15
            else:
                num_e_15 += len(pred_list[0:15])

            num_s += len(label_list)

        p_5, r_5, f_5 = get_PRF(num_c_5, num_e_5, num_s)
        print_PRF(p_5, r_5, f_5, 5)
        p_10, r_10, f_10 = get_PRF(num_c_10, num_e_10, num_s)
        print_PRF(p_10, r_10, f_10, 10)
        p_15, r_15, f_15 = get_PRF(num_c_15, num_e_15, num_s)
        print_PRF(p_15, r_15, f_15, 15)

        f_5_scores.append(f_5*100)
        f_10_scores.append(f_10*100)
        f_15_scores.append(f_15*100)

        logging.info('---------------------')

    avg_f_5 = sum(f_5_scores) / len(f_5_scores)
    logging.info("F1@5 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_5_scores))
    logging.info("Average F1@5 Score: " + f"{avg_f_5:.2f}")

    avg_f_10 = sum(f_10_scores) / len(f_10_scores)
    logging.info("F1@10 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_10_scores))
    logging.info("Average F1@10 Score: " + f"{avg_f_10:.2f}")

    avg_f_15 = sum(f_15_scores) / len(f_15_scores)
    logging.info("F1@15 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_15_scores))
    logging.info("Average F1@15 Score: " + f"{avg_f_15:.2f}")
  
