# train
python cvpr20-scatter-text-recognizer/train.py --train_data data/train --valid_data data/valid --select_data RGB --batch_ratio 1.0 --sensitive

python cvpr20-scatter-text-recognizer/train.py --train_data data/train --valid_data data/valid --select_data RGB --batch_ratio 1.0 --sensitive --worker 0 --rgb

# demo 
python cvpr20-scatter-text-recognizer/demo.py --saved_model saved_models/best_accuracy_99.55357142857143.pth --sensitive --image_folder ./data/test