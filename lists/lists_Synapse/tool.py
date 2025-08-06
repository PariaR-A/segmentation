import glob

def write_names():
    # Process test files
    test_files = glob.glob('/content/segmentation/datasets/Synapse/test_vol_h5/*.npz')  # or *.np
    with open('/content/segmentation/lists/lists_Synapse/test_vol.txt', 'w') as f:
        for file in test_files:
            name = file.split('/')[-1][:-4] + '\n'  # removes extension and adds newline
            f.write(name)
    
    # Process training files
    train_files = glob.glob('/content/segmentation/datasets/Synapse/train_npz/*.npz')  # or *.np
    with open('/content/segmentation/lists/lists_Synapse/train.txt', 'w') as f:
        for file in train_files:
            name = file.split('/')[-1][:-4] + '\n'  # removes extension and adds newline
            f.write(name)

    print("Finished creating both test.txt and train.txt!")

write_names()