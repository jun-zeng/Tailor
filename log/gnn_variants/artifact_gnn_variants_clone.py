import subprocess

# Copy this file to `cpgnn` directory and run it
# Please ensure you have oj_clone_encoding, oj_classification_encoding, and bcb_clone_encoding under Tailor/cpgnn/data

args = [
        # code clone detection on the OJ dataset
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --agg_type none --report clone_oj_none',
        
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --agg_type lightgcn --report clone_oj_lightgcn',
        
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --agg_type gcn --report clone_oj_gcn',

        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512 --agg_type kgat --report clone_oj_kgat'

        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --agg_type ggnn --report clone_oj_ggnn',
        ]

def main():
    for arg in args:
        p = subprocess.Popen(arg, shell=True)
        p.wait()

if __name__ == '__main__':
    main()
