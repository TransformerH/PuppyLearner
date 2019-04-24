import os
from PuppyDetection import extract_object
from torch.utils.data import DataLoader
from load import load_datasets

batch_size = 4

# Load dataset
train_data, test_data, classes = load_datasets()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#################################  write boxes list from file  #################################
save_box_path = os.getcwd() + "/boxes/"
print(save_box_path)
try:
    os.mkdir(save_box_path)
except OSError:
    print("Can't create folder")

for batch_i, data in enumerate(train_loader):
    original_images, labels, fileNames = data
    dog_box, dog_head_box = extract_object(original_images)
    # save to the file
    for i in range(len(fileNames)):
        save_fileName = fileNames[i].split('.')[0]
        print("save_fileName: " + save_fileName)
        save_file_path = os.path.join(save_box_path, save_fileName + '.txt')
        with open(save_file_path, 'w') as f:
            for item in dog_box[i]:
                f.write("%s " % item)
            f.write("\n")
            for item in dog_head_box[i]:
                f.write("%s " % item)
        f.close()
#################################  write boxes list from file  #################################

print("write files done")