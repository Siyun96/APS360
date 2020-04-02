# C:\Windows\System32\bash.exe -c "./tuning.sh"

echo "Running Hyperparameter Tuning..."

# Default parameters
echo "Running with default hyperparameters..."
C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --batch_size=32 --learning_rate=0.001 --num_epochs=20 --cnn classifier

echo "Trying different learning rates..."
for LR in 0.0005, 0.005; do
    echo "Changing learning rate to $LR..."
    C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --batch_size=32 --learning_rate=$LR --num_epochs=20 --cnn classifier
done

echo "Trying different batch sizes..."
for BS in 16, 64; do
    echo "Changing batch size to $BS..."
    C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --batch_size=$BS --learning_rate=0.001 --num_epochs=20 --cnn classifier
done

echo "Trying different architectures..."
echo "Default architecture: classifier"
for feature1 in 50, 100; do
    echo "Changing feature1 to $feature1..."
    for feature2 in 100, 150; do
        echo "Changing feature2 to $feature2..."
        C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --cnn classifier $feature1 $feature2 $feature2 200 100
    done
done
for hidden1 in 400, 100; do
    echo "Changing hidden1 to $hidden1..."
    C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --cnn classifier 50 100 100 $hidden1 $((hidden1 / 2))
done


echo "Model A..."
C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --cnn a

echo "Model B..."
C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --cnn b

echo "Model C..."
C:/Users/li_si/Anaconda3/envs/Capstone/python.exe c:/Users/li_si/Documents/Siyun/EngSci_Year4/Winter/APS360/Project/Code/APS360/train.py --cnn c

