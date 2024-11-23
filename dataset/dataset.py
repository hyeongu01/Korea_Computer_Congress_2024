from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from PIL import Image
import os

'''

- customDataset, transform 선언
- add sault and pepper noise 함수 선언
- 데이터셋 png이미지로 저장하는 함수 선언

+ 뒷부분에 사용법 추가

'''

class customDataset(Dataset):
    def __init__(self, data_path, train, noise_prob=0.0, transform=None):
        self.data = datasets.MNIST(root=data_path, train=train, download=True)
        self.transform = transform
        self.noise_prob = noise_prob
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, _ = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
        noisy_img = add_salt_and_pepper_noise(image, self.noise_prob)
            
        return noisy_img, image

transform = transforms.Compose([
    transforms.ToTensor()
])


def add_salt_and_pepper_noise(image, prob):
    """
    이미지에 소금-후추 노이즈를 추가하는 함수
    image: 입력 이미지 (numpy 배열)
    prob: 노이즈가 발생할 확률 (0에서 1 사이 값, 예: 0.02는 2% 확률로 노이즈 추가)
    """
    output = np.copy(image)
    # 이미지 크기
    total_pixels = image.shape[-1] * image.shape[-2]

    
    # 소금(255) 노이즈를 추가할 픽셀의 수
    num_salt = int(total_pixels * prob / 2)
    # 후추(0) 노이즈를 추가할 픽셀의 수
    num_pepper = int(total_pixels * prob / 2)
    # 소금 노이즈 추가 (255)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[1:]]
    output[0][tuple(coords)] = 1
    
    # 후추 노이즈 추가 (0)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[1:]]
    output[0][tuple(coords)] = 0
    
    return torch.Tensor(output)

def save_dataset_as_images(dataset, output_dir):
    noisy_dir = os.path.join(output_dir, 'noisy')  # noisy 이미지 저장 폴더
    original_dir = os.path.join(output_dir, 'original')  # original 이미지 저장 폴더
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    for idx, (noisy_img, original_img) in enumerate(dataset):
        # 텐서를 numpy 배열로 변환
        noisy_img_np = (noisy_img.numpy() * 255).astype(np.uint8).squeeze()
        original_img_np = (original_img.numpy() * 255).astype(np.uint8).squeeze()

        # noisy 이미지 저장
        noisy_img_pil = Image.fromarray(noisy_img_np, mode='L')
        noisy_img_pil.save(os.path.join(noisy_dir, f"noisy_{idx}.png"))
        
        # original 이미지 저장
        original_img_pil = Image.fromarray(original_img_np, mode='L')
        original_img_pil.save(os.path.join(original_dir, f"original_{idx}.png"))

        # 진행 상황 출력
        if idx % 1000 == 0:
            print(f"Saved {idx} images...")

    print(f"All images saved to {output_dir}")


if __name__ == "__main__":
    data_path = './mnist'
    train_dataset = customDataset(data_path, True, transform=transform)
    test_dataset = customDataset(data_path, False, transform=transform)

    save_dataset_as_images(test_dataset, "./mnist_test_png_2")

    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)