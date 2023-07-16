from fastai.vision import *
from fastai.callback import *

PATH = '../images_ecoli/'
EPOCHS = 50

############################################
# 01 train and test original model
############################################
tfms = get_transforms(do_flip=True,flip_vert=False,max_rotate=0,max_zoom=1,max_lighting=None,max_warp=None,p_affine=0,p_lighting=0)
data = ImageDataBunch.from_folder(PATH,ds_tfms=tfms, bs=32)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(EPOCHS,max_lr=0.0001,callbacks=[callbacks.SaveModelCallback(learn,every='improvement',monitor='accuracy',name='original_ecoli_model')])

data = ImageDataBunch.from_folder(PATH, test='test_operons/')
output = open('operon_pegs.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Operons':
        output.write(fname + '\n')
    c += 1
output.close()

data = ImageDataBunch.from_folder(PATH, test='test_noperons/')
output = open('noperon_pegs.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Noperons':
        output.write(fname + '\n')
    c += 1
output.close()

############################################
# 02 train and test ablation models
############################################

# remove horizontal flip
tfms = get_transforms(do_flip=False,flip_vert=False,max_rotate=0,max_zoom=1,max_lighting=None,max_warp=None,p_affine=0,p_lighting=0)
data = ImageDataBunch.from_folder(PATH,ds_tfms=tfms, bs=32)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(EPOCHS,max_lr=0.0001,callbacks=[callbacks.SaveModelCallback(learn,every='improvement',monitor='accuracy',name='no_hflip_ecoli_model')])

data = ImageDataBunch.from_folder(PATH, test='test_operons/')
output = open('operon_pegs_no_hflp.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Operons':
        output.write(fname + '\n')
    c += 1
output.close()

data = ImageDataBunch.from_folder(PATH, test='test_noperons/')
output = open('noperon_pegs_no_hflp.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Noperons':
        output.write(fname + '\n')
    c +=1 
output.close()

# remove zoom
tfms = get_transforms(do_flip=True,flip_vert=False,max_rotate=0,max_zoom=0,max_lighting=None,max_warp=None,p_affine=0,p_lighting=0)
data = ImageDataBunch.from_folder(PATH,ds_tfms=tfms, bs=32)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(EPOCHS,max_lr=0.0001,callbacks=[callbacks.SaveModelCallback(learn,every='improvement',monitor='accuracy',name='no_zoom_ecoli_model')])

data = ImageDataBunch.from_folder(PATH, test='test_operons/')
output = open('operon_pegs_no_zoom.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Operons':
        output.write(fname + '\n')
    c += 1
output.close()

data = ImageDataBunch.from_folder(PATH, test='test_noperons/')
output = open('noperon_pegs_no_zoom.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Noperons':
        output.write(fname + '\n')
    c += 1
output.close()

# remove zoom and horizontail flip
tfms = get_transforms(do_flip=False,flip_vert=False,max_rotate=0,max_zoom=0,max_lighting=None,max_warp=None,p_affine=0,p_lighting=0)
data = ImageDataBunch.from_folder(PATH,ds_tfms=tfms, bs=32)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(EPOCHS,max_lr=0.0001,callbacks=[callbacks.SaveModelCallback(learn,every='improvement',monitor='accuracy',name='no_zoom_no_hflip_ecoli_model')])

data = ImageDataBunch.from_folder(PATH, test='test_operons/')
output = open('operon_pegs_no_zoom.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Operons':
        output.write(fname + '\n')
    c += 1
output.close()

data = ImageDataBunch.from_folder(PATH, test='test_noperons/')
output = open('noperon_pegs_no_zoom.txt', 'w')
c = 0
for i in data.test_ds:
    p = learn.predict(i[0])
    filename = str(data.test_ds.items[c]).split('/')[-1]
    fname = filename.split('_')[0]
    if str(p[0]) == 'Noperons':
        output.write(fname + '\n')
    c += 1
output.close()
