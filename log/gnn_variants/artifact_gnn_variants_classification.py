import subprocess

# Copy this file to `cpgnn` directory and run it
# Please ensure you have oj_clone_encoding, oj_classification_encoding, and bcb_clone_encoding under Tailor/cpgnn/data

args = [
        # source clone classification on the OJ dataset
        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --agg_type none --report classification_oj_none',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --agg_type lightgcn --report classification_oj_lightgcn',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --agg_type gcn --report classification_oj_gcn',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_classification 384 --agg_type kgat --report classification_oj_kgat',
        
        # Due to the limited memory of our GPUs, we train GGNN using CPUs, which takes more than two weeks (You can need to hardcode cpgnn/model/SGL.py to achieve this functionality)
        # 'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --agg_type ggnn --report classification_oj_ggnn_cpu',
        ]

def main():
    for arg in args:
        p = subprocess.Popen(arg, shell=True)
        p.wait()

if __name__ == '__main__':
    main()
