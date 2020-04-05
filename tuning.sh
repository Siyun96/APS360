# C:\Windows\System32\bash.exe -c "./tuning.sh"

echo "Running Hyperparameter Tuning..."

# 04-03 experiments
echo "Running simpler architecture..."
echo "Model d"
for feature in 25 30 35; do
    echo "Changing number of features to $feature $((feature * 2))"
    python /home/siyun/MyUnityProjects/CNN/APS360/train.py --batch_size=32 --learning_rate=0.001 --num_epochs=15 --cnn 4 $feature $((feature * 2)) $((feature * 2)) 200
done

# for LR in 0.001 0.002 0.003; do
#     echo "Changing learning rate to $LR"
#     python /home/siyun/MyUnityProjects/CNN/APS360/train.py --batch_size=32 --learning_rate=$LR --num_epochs=15 --cnn 4
# done


# 04-01 experiments
# Default parameters
# echo "Running with default hyperparameters..."
# python /home/siyun/MyUnityProjects/CNN/APS360/train.py --batch_size=32 --learning_rate=0.001 --num_epochs=10 --cnn 0

# echo "Trying different learning rates..."
# for LR in 0.0005 0.005; do
#     echo "Changing learning rate to $LR..."
#     python /home/siyun/MyUnityProjects/CNN/APS360/train.py --batch_size=32 --learning_rate=$LR --num_epochs=10 --cnn 0
# done

# echo "Trying different batch sizes..."
# for BS in 16 64; do
#     echo "Changing batch size to $BS..."
#     python /home/siyun/MyUnityProjects/CNN/APS360/train.py --batch_size=$BS --learning_rate=0.001 --num_epochs=10 --cnn 0
# done

# echo "Trying different architectures..."
# echo "Default architecture: classifier"
# for feature1 in 50 100; do
#     echo "Changing feature1 to $feature1..."
#     for feature2 in 100 150; do
#         echo "Changing feature2 to $feature2..."
#         python /home/siyun/MyUnityProjects/CNN/APS360/train.py --cnn 0 $feature1 $feature2 $feature2 200 100
#     done
# done
# for hidden1 in 400 100; do
#     echo "Changing hidden1 to $hidden1..."
#     python /home/siyun/MyUnityProjects/CNN/APS360/train.py --cnn 0 50 100 100 $hidden1 $((hidden1 / 2))
# done


# echo "Model A..."
# python /home/siyun/MyUnityProjects/CNN/APS360/train.py --cnn 1

# echo "Model B..."
# python /home/siyun/MyUnityProjects/CNN/APS360/train.py --cnn 2

# echo "Model C..."
# python /home/siyun/MyUnityProjects/CNN/APS360/train.py --cnn 3

