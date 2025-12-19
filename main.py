import argparse
from classify_embeddings import load_svm_gpu, train_svm_gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPMI-SVC Model Execution Script. Requires a MODEL_NAME followed by --train or --test."
    )

    parser.add_argument(
        'modelname',
        type=str,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--train', 
        action='store_true',
    )
    group.add_argument(
        '--test', 
        action='store_true',
        help='Run the testing/comparison workflow.'
    )

    args = parser.parse_args()
    modelname = args.modelname 

    if args.train:
        train_svm_gpu(modelname=modelname) 
        print(f"SVC Training complete. Model name: {modelname}")

    elif args.test:
        load_svm_gpu(modelname=modelname) 
        print(f"Testing/Comparison complete using model: {modelname}")