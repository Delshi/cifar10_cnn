The **cifar-10** CNN classifier with **~92%** accuracy.
It uses some augmentation of images, contains logging of lots of metrics such as projector, PR-curves, etc.

# How to use
## Installation
1. Run requirements.bat and then start model.py
2. Unpack archives in **logs** folder
## Usage
1. To start learning, run start_tb.bat and then run model.py
2. Before learning delete logs directory, then run model.py
3. To see model predictions after learning, run model.py
4. DATA directory contains results of my attempt of learning
5. Logs directory contains all the results of learning
# Results
## Graphs
Epoch Acc and Loss<br/>
<img width="1200" height="400" alt="Acc_Loss Plot" src="https://github.com/user-attachments/assets/a353aa49-51fe-4af6-a1c9-a66e5e679b0a" /><br/>
Epoch LR<br/>
<img width="2153" height="477" alt="Epoch Learning Rate" src="https://github.com/user-attachments/assets/8f31644e-fcd7-4d61-bb91-bdbc14390cce" /><br/>
## Confusion Matrix
Epoch 3<br/>
<img width="800" height="800" alt="Confusion Matrix epoch 3" src="https://github.com/user-attachments/assets/1363ce55-fa53-43d2-82cd-136b86290b0b" /><br/>
Epoch 84<br/>
<img width="800" height="800" alt="Confusion Matrix epoch 84" src="https://github.com/user-attachments/assets/dbd4adf5-a737-41de-80b3-2d8bf40b946e" /><br/>
## PR Curves
Automobile Epoch 1<br/>
<img width="800" height="800" alt="PR Curve automobile epoch 1" src="https://github.com/user-attachments/assets/2df34bc7-a7e1-4b8f-9cc6-cdd60147a06b" /><br/>
Automobile Epoch 84<br/>
<img width="800" height="800" alt="PR Curve automobile epoch 84" src="https://github.com/user-attachments/assets/aa62449d-1b96-499b-b165-9f8d9841935e" /><br/>
Cat Epoch 1<br/>
<img width="800" height="800" alt="PR Curve cat epoch 1" src="https://github.com/user-attachments/assets/592b1157-6d7d-46c5-82aa-3ebf29cef8ab" /><br/>
Cat Epoch 84<br/>
<img width="800" height="800" alt="PR Curve cat epoch 84" src="https://github.com/user-attachments/assets/478bc569-5082-42ee-82cf-91d890b7c55b" /><br/>
## Projector
PCA<br/>
<img width="2559" height="1439" alt="PCA" src="https://github.com/user-attachments/assets/71d1bc40-7266-48ff-ab9c-6cd71ebbb7f9" /><br/>
UMAP 2D<br/>
<img width="2559" height="1439" alt="UMAP 2D" src="https://github.com/user-attachments/assets/0024cc47-3ea2-439c-a656-fb7dbbd9ab97" /><br/>
UMAP 3D<br/>
<img width="2559" height="1439" alt="UMAP 3D" src="https://github.com/user-attachments/assets/e1595db5-f2e8-48f2-a0f6-16cbbd0f4d89" /><br/>
## Histograms
Gradients Beta<br/>
<img width="682" height="516" alt="Gradients Beta" src="https://github.com/user-attachments/assets/50028162-497a-42c7-963c-89ffda85f5f8" /><br/>
Gradients Bias<br/>
<img width="688" height="515" alt="Gradients Bias" src="https://github.com/user-attachments/assets/90cb3387-5ff0-43de-89a5-e81896bbe193" /><br/>
Main Kernel<br/>
<img width="680" height="513" alt="Kernel" src="https://github.com/user-attachments/assets/03cbef16-a176-4032-8fe2-4dddb0903f5c" /><br/>
Moving Mean<br/>
<img width="689" height="518" alt="Moving Mean" src="https://github.com/user-attachments/assets/6b5d47bc-5f01-4245-855d-285bc0143dc0" /><br/>
## Distributions
Gradients Gamma<br/>
<img width="717" height="512" alt="Gradients Gamma" src="https://github.com/user-attachments/assets/906058d1-cd8f-4fd8-9077-65fd2809aed6" /><br/>
Gradients Kernel<br/>
<img width="723" height="507" alt="Gradients Kernel" src="https://github.com/user-attachments/assets/96de9c12-8c3f-4b54-b359-991df2988930" /><br/>
Main Kernel<br/>
<img width="717" height="517" alt="Kernel" src="https://github.com/user-attachments/assets/f57189b0-5fed-4122-90e0-325b3294a371" /><br/>
Moving Mean<br/>
<img width="717" height="513" alt="Moving Mean" src="https://github.com/user-attachments/assets/92a0b580-71f7-4dfe-abae-8a9131699410" /><br/>
Moving Variance<br/>
<img width="715" height="512" alt="Moving Variance" src="https://github.com/user-attachments/assets/a0b52368-b22d-4ddd-81b8-c3227dba1781" /><br/>
### More in a DATA directory




