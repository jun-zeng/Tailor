import subprocess

# Copy this file to `cpgnn` directory and run it
# Please ensure you have oj_clone_encoding, oj_classification_encoding, and bcb_clone_encoding under Tailor/cpgnn/data

args = [
        # code clone detection on the OJ dataset
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512 --embed_init token --report clone_oj_token',

        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512 --embed_init type --report clone_oj_type',

        # code clone detection on the BCB dataset
        'python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384  --embed_init token --report clone_bcb_token',

        'python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384  --embed_init type --report clone_bcb_type',

        # code classification on the OJ dataset
        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --embed_init token --report classification_oj_token',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --embed_init type --report classification_oj_type',
        ]

def main():
    for arg in args:
        p = subprocess.Popen(arg, shell=True)
        p.wait()

if __name__ == '__main__':
    main()
