import argparse
import time
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models.autoencoder import load_model
from dataset.dataset import customDataset, transform, save_dataset_as_images

'''
Useage: 
    python test.py --pretrained_path {*.pth} --data_path {data_path(mnist)} --output_dir {directory for output images}

    + optional:
        --batch_size {default=512}
'''

def get_args():
    """
    Returns:
        args (argparse.Namespace): 명령줄 인자 객체.
            - pretrained_path: 훈련된 모델 경로.
            - data_path: 데이터셋 경로.
            - output_dir: 출력 디렉터리 경로.
    """
    parser = argparse.ArgumentParser(description="Parse arguments for model loading and data processing.")
    
    # 명령줄 인자 추가
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model file (.pth)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where results will be saved."
    )

    # 옵션값
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
    
    # 명령줄 인자 파싱
    args = parser.parse_args()

    return args

def test(model, test_loader, device, results_dir):
    total_samples = 0
    total_time = 0.0

    model.eval()

    with torch.no_grad():
        start_time = time.time()

        for batch_idx, (noisy_imgs, original_imgs) in enumerate(test_loader):
            noisy_imgs, original_imgs = noisy_imgs.to(device), original_imgs.to(device)

            # 모델 출력 (복원된 이미지)
            restored_imgs = model(noisy_imgs)

            # 배치 처리 시간 측정
            batch_time = time.time() - start_time
            total_time += batch_time

            # 배치 내 샘플 저장
            for i in range(restored_imgs.size(0)):
                idx = batch_idx * args.batch_size + i
                restored_img_np = (restored_imgs[i].cpu().numpy() * 255).astype(np.uint8).squeeze()
                original_img_np = (original_imgs[i].cpu().numpy() * 255).astype(np.uint8).squeeze()

                restored_pil = Image.fromarray(restored_img_np, mode='L')
                restored_pil.save(os.path.join(results_dir, f"restored_{idx}.png"))

            # 배치 내 샘플 수 누적
            total_samples += restored_imgs.size(0)
            print(f"Batch {batch_idx + 1} processed, Batch time: {batch_time:.4f} seconds.")

            start_time = time.time()
    avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
    return total_samples, total_time, avg_time_per_sample
    


def main(args):
    # 데이터 로드 및 테스트 데이터를 이미지로 저장
    data_path = args.data_path
    output_dir = args.output_dir

    test_dataset = customDataset(data_path, False, transform)
    print("=== Original & noisy image saving ===")
    save_dataset_as_images(test_dataset, output_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


    # 테스트 파라메터 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.pretrained_path).to(device)

    results_dir = os.path.join(output_dir, "restored")
    os.makedirs(results_dir, exist_ok=True)

    # 테스트 후 저장
    print("\n=== Starting model evaluation ===")
    total_samples, total_time, avg_time_per_sample = test(
        model, test_loader, device, results_dir
        )
    
    # 결과 출력
    print("\n=== Evaluation done ===")
    print(f"Total samples processed: {total_samples} samples")
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.6f} seconds/sample")
    print("===========================")
    print(f"\nResults are saved at {output_dir}")


if __name__ == "__main__":
    args = get_args()
    main(args)