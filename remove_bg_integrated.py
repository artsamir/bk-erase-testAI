import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from U2Net.data_loader import RescaleT, ToTensorLab, SalObjDataset
from U2Net.model import U2NET, U2NETP
import glob

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = os.path.basename(image_name)
    image = Image.open(image_name)
    imo = im.resize((image.size[0], image.size[1]), resample=Image.Resampling.BILINEAR)
    imo.save(os.path.join(d_dir, os.path.splitext(img_name)[0] + '.png'))

def remove_background_integrated(image_path, output_path, model_name='u2net'):
    model_dir = os.path.join('U2Net', 'saved_models', model_name + '.pth')
    prediction_dir = 'test_results'
    os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = [image_path]

    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)]))
    test_salobj_dataloader = torch.utils.data.DataLoader(test_salobj_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         num_workers=1)

    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        save_output(image_path, pred, prediction_dir)

        mask_path = os.path.join(prediction_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        image.putalpha(mask)
        image.save(output_path)
        print(f"Background removed and saved to: {output_path}")

# Process all images in the test_images directory
image_dir = "input_images"  # Replace with your input image directory
output_dir = "output_bk_remove" #Replace with your output image directory.

os.makedirs(output_dir, exist_ok=True)

image_files = glob.glob(os.path.join(image_dir, '*')) #get all files in the image directory.

for image_file in image_files:
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_name = os.path.basename(image_file)
        output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_removed.png")
        remove_background_integrated(image_file, output_path)