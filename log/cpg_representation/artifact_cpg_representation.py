import subprocess

# Copy this file to `cpgnn` directory and run it
# Please ensure you have oj_clone_encoding, oj_classification_encoding, and bcb_clone_encoding under Tailor/cpgnn/data

args = [
        # code clone detection on the OJ dataset
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --cpg_no_cfg --report clone_oj_no_cfg',
        
        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --cpg_no_dfg --report clone_oj_no_dfg',

        'python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512  --cpg_no_cfg --cpg_no_dfg --report clone_oj_no_cfg_dfg',

        # code clone detection on the BCB dataset
        'python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384 --cpg_no_cfg  --report clone_bcb_no_cfg',

        'python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384 --cpg_no_dfg  --report clone_bcb_no_dfg',

        'python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384 --cpg_no_cfg --cpg_no_dfg  --report clone_bcb_no_cfg_dfg',

        # code classification on the OJ dataset
        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --cpg_no_cfg --report classification_oj_no_cfg',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --cpg_no_dfg --report classification_oj_no_dfg',

        'python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --cpg_no_cfg --cpg_no_dfg --report classification_oj_no_cfg_dfg',
        ]

def main():
    for arg in args:
        p = subprocess.Popen(arg, shell=True)
        p.wait()

if __name__ == '__main__':
    main()
