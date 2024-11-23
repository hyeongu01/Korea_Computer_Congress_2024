import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import customDataset, transform
from models.autoencoder import load_model

'''
Useage: 
    python train.py --data_dir {data_dir} --model_save_dir {model_save_dir}

    + optional:
        --batch_size {default=512}
        --epochs {default=50}
        --save_epoch {default=5}
        --leraning_rate {default=0.001}
        --noise_prob {default=0.4}
'''

def get_args():
    """
    Returns:
        args (argparse.Namespace): 명령줄 인자 객체.
            - data_dir: 학습에 사용할 데이터셋 경로.
            - model_save_dir: 학습된 모델을 저장할 디렉터리 경로.
            - batch_size: 학습 시 사용할 배치 크기 (기본값: 512).
            - epochs: 학습 에포크 수 (기본값: 20).
    """
    parser = argparse.ArgumentParser(description="Parse arguments for model training.")
    
    # 명령줄 인자 추가
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="data dir for training"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=True,
        help="dir for model save"
    )

    # 옵션값
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--save_epoch", type=int, default=5, help="Save the model every N epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--noise_prob', type=float, default=0.4, help='Noise probability for training')

    # 명령줄 인자 파싱
    args = parser.parse_args()

    return args

def train(model, train_loader, criterion, optimizer, num_epochs, device, noise_prob, save_epoch=5, model_path="./model_checkpoint/"):
    model.to(device)
    checkpoint_path = "checkpoint.pth"
    average_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        with tqdm(train_loader, unit='batch', leave=False) as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                gen_img = model(x)

                loss = criterion(gen_img, y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
        
        average_loss = total_loss / len(train_loader)
        average_loss_list.append(average_loss)
        print(f"Epochg {epoch + 1}/{num_epochs}, Loss: {average_loss:.12f}")

        # 가장 최근의 체크포인트
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss
        }, os.path.join(model_path, checkpoint_path))

        # save_epoch 마다 체크포인트 저장
        if (epoch + 1) % save_epoch == 0:
            save_path = os.path.join(model_path, f"epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss
            }, save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")
    
    loss_graph_path = os.path.join(model_path, "loss_plot.png")
    save_loss_plot(average_loss_list, loss_graph_path, noise_prob)
    print(f"Loss plot saved to {save_path}")

def save_loss_plot(loss_list, save_path, noise_prob):
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', color='b', label='Loss')
    plt.title(f"Training Loss p={noise_prob}", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # 그래프 저장
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    


def main(args):
    # 학습 데이터 로드
    data_path = args.data_dir
    train_dataset = customDataset(data_path, True, args.noise_prob, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 학습 파라메터 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    lr = args.learning_rate

    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    creterion = torch.nn.MSELoss()
    
    save_epoch = args.save_epoch
    model_save_dir = args.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

     # 학습 파라미터 출력
    print("=== Training Parameters ===")
    print(f"Device: {device}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model save directory: {model_save_dir}")
    print(f"Save model every {save_epoch} epochs")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {creterion}")
    print("===========================")

    # 학습
    print("\n=== Training start ===")
    train(model, train_loader, creterion, optimizer, epochs, device, args.noise_prob, save_epoch, model_save_dir)
    print("\n=== Training is completed ===")
    print(f"Model and Loss Graph is saved at {model_save_dir}")


if __name__ == "__main__":
    args = get_args()
    main(args)