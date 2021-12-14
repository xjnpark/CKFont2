#!/usr/bin/env python

import argparse
import glob
import io
import os
import shutil


from PIL import Image, ImageFont, ImageDraw

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_SPLIT_FILE = os.path.join(SCRIPT_PATH,
                                  '../tgt-split-image-data/images/*.jpeg')
DEFAULT_LABEL_SPLIT_NO_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2000-common-hangul-split_no.txt')
DEFAULT_SOURCE_FONTS_DIR = os.path.join(SCRIPT_PATH, '../src-image-data/images/*.jpeg')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../tgt_font')
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2000-common-hangul.txt')
DEFAULT_FONTS_IMG_DIR = os.path.join(SCRIPT_PATH, '../tgt-image-data/images/*.jpeg')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../combine-image-data')

def remove_dir(path):
    #param <path> could either be relative or absolute. 
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))



def generate_comb_images(label_split_file, label_split_no_file, src_fonts_dir, fonts_img_dir, fonts_dir, output_dir):
    # Width and height of the resulting image.
    comb_width = 1280
    comb_height = 256
    comb = Image.new("L",(comb_width, comb_height))

    # imglist=[]
    # images_tgt_split = glob.glob(label_split_file)
    # images_tgt = glob.glob(fonts_img_dir)
    # images_src = glob.glob(src_fonts_dir)
     
    images_tgt_split = sorted(glob.glob(label_split_file))
    images_tgt = sorted(glob.glob(fonts_img_dir))
    images_src = sorted(glob.glob(src_fonts_dir))

    f = open(label_split_no_file, 'r')
    images_split_no = f.read()
    f.close()

    # with io.open(label_file, 'r', encoding='utf-8') as f:
    #     labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    total_count = 0
    font_count = 0
    print('total number of fonts are ', len(fonts))

    # idx = iter(range(len(images_tgt)))
    k=0
    im_3 = Image.new("L",(256, 256)) 

    for id, i in enumerate(range(0,len(images_tgt_split),2)): # split img 먼저 combine
        total_count += 1
        # j= next(idx)
        f1=os.path.basename(images_tgt[id]).split('.')[0]
        
        # f1=os.path.splitext(os.path.basename(images_tgt[id]))[0]
        # ftype=int(f1[0])
        # f2=int(f1[2:])
        im_src= Image.open(images_src[id])
        im_0 = Image.open(images_tgt[id])
        im_1 = Image.open(images_tgt_split[i+k])
        im_2 = Image.open(images_tgt_split[i+k+1])
        #print("i=",i, "k=",k, "i+k=", i+k,"id = j = ",id, ' len_split_no', int(len(images_split_no)))

        comb.paste(im_src, box=(0, 0))
        comb.paste(im_0, box=(256, 0))
        comb.paste(im_1, box=(512, 0))
        comb.paste(im_2, box=(768, 0))
        comb.paste(im_3, box=(1024, 0))
        #comb.crop(box=(0,0, 1024, 256))

        # comb.paste(im_3, box=(1024, 0))

        if (int(images_split_no[id])) >= 3  :
        # if (int(images_split_no[id])) >= 3 and (i+k+2 <= len(images_tgt_split)) :
            im_3 = Image.open(images_tgt_split[i+k+2])
            comb.paste(im_3, box=(1024, 0))
            k += 1 
            file_string = '{}.jpeg'.format(f1)
            file_path = os.path.join(image_dir, file_string)
            comb.save(file_path, 'JPEG')
            im_3 = Image.new("L",(256, 256))

            # comb=comb.crop(box=(0,0, 1024, 256))

        else :
            file_string = '{}.jpeg'.format(f1)
            file_path = os.path.join(image_dir, file_string)
            # comb_temp=comb.crop(box=(0,0,1024,256))

            comb.save(file_path, 'JPEG')

        # comb_empty=comb.crop(box=(1024,0,1280,256))

        if id+1 == int(len(images_split_no)):
            break

    print('Finished generating {} images.'.format(total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label-split-file', type=str, dest='label_split_file',
                        default=DEFAULT_LABEL_SPLIT_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--label-split-no-file', type=str, dest='label_split_no_file',
                        default=DEFAULT_LABEL_SPLIT_NO_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--src-font-dir', type=str, dest='src_fonts_dir',
                        default=DEFAULT_SOURCE_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--font-img-dir', type=str, dest='fonts_img_dir',
                        default=DEFAULT_FONTS_IMG_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images.')
    args = parser.parse_args()

    generate_comb_images(args.label_split_file, args.label_split_no_file, args.src_fonts_dir, args.fonts_img_dir, args.fonts_dir, args.output_dir)
   
   # remove the src and target directories
    # src_head, _ = os.path.split(args.src_fonts_dir)
    # tgt_head, _ = os.path.split(args.fonts_img_dir)
    # print("Removing the directories")
    # remove_dir(src_head)
    # remove_dir(tgt_head)
    print("*** DONE ***") 