import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_difference_image(image1_path,image2_path,output_path):
    # 读取图像
    if isinstance(image1_path,str): 
        img1 = cv2.imread(image1_path)
    else:
        img1 = np.array(image1_path)
    
    if isinstance(image2_path,str):
        img2 = c2.imread(image2_path)
    else:
        img2 = np.array(image2_path)

    # img2 = cv2.imread(image2_path)

    # 确保两个图像具有相同的尺寸
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算差异图像并归一化以便可视化
    diff_img = cv2.absdiff(img1, img2)
    diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

    # 打印MSE值
    print(f'MSE: {mse}')

    # 显示原始图像和差异图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
    plt.title('Difference Image')
    plt.axis('off')

    plt.suptitle(f'MSE: {mse:.2f}')

    # 保存图片
    plt.savefig(output_path)
    plt.close()  # 关闭图像以释放内存

    return mse


if __name__=="__main__":
    # TODO:scheduler写出去

    # image1_path_list = ["images/output.png"]*3
    # image2_path_list = ["images/sd3_noise_to_x0_20.png","images/sd3_noise_to_x0_200.png","images/sd3_noise_to_x0_400.png"]
    # output_path_list = ["images/difference_images_20.png","images/difference_images_200.png","images/difference_images_400.png"]
    
    # for i in range(3):
    #     save_difference_image(image1_path_list[i],image2_path_list[i],output_path_list[i])

    # image1_path_list = ["images/output.png"]*3

    # image1_path_list = ["images/sd3_noise_to_x0_20.png","images/sd3_noise_to_x0_20.png","images/sd3_noise_to_x0_200.png"]
    # image2_path_list = ["images/sd3_noise_to_x0_20.png","images/sd3_noise_to_x0_200.png","images/sd3_noise_to_x0_400.png"]
    # output_path_list = ["images/difference_images_20.png","images/difference_images_200.png","images/difference_images_400.png"]
    
    # for i in range(3):
    #     save_difference_image(image1_path_list[i],image2_path_list[i],output_path_list[i])


    image1_path_list = ["images_sd/sd_noise_to_x0_50_.png"]
    image2_path_list = ["images_sd/sd_noise_to_x0_50.png"]
    output_path_list = ["images_sd/difference_images_50.png"]
    for i in range(1):
        save_difference_image(image1_path_list[i],image2_path_list[i],output_path_list[i])
