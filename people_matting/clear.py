import shutil, os


# clean and rebuild the image folders
input_folder = '../data/test/input'
# if os.path.exists(input_folder):
#     shutil.rmtree(input_folder)
# os.makedirs(input_folder)

output_folder = '../data/test/input'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


