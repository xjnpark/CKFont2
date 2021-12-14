#!/usr/bin/env python

import argparse
import glob
import io
import os

from PIL import Image, ImageFont, ImageDraw

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2000-common-hangul-split.txt')
DEFAULT_LABEL_FILE_2 = os.path.join(SCRIPT_PATH,
                                  '../labels/2000-common-hangul-split_no.txt')
DEFAULT_TARGET_FONTS_DIR = os.path.join(SCRIPT_PATH, '../tgt_font')
DEFAULT_SOURCE_FONTS_DIR = os.path.join(SCRIPT_PATH, '../src_font')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../src-split-image-data')

# Width and height of the resulting image.
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
# fname = '../labels/256-common-hangul-split_no.txt'

def generate_hangul_images(label_file, label_file_2, target_fonts_dir, src_fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory.
    # """
    labels_s=[]
    labels_s_no=[]
    with io.open(label_file, 'rt', encoding='utf-8') as f:
        labels = f.read().splitlines()
        #print(len(labels))
        for i in range(len(labels)):
            labels_s.append(labels[i][0])
            labels_s.append(labels[i][1])
            #print(len(labels[i]))

            if (len(labels[i])) >= 3 :
                labels_s.append(labels[i][2])
            labels_s_no.append(len(labels[i]))
            #print(labels_s_no[i]) 

    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(src_fonts_dir, '*.ttf'))

    # Get a list of the  target fonts.
    target_fonts = glob.glob(os.path.join(target_fonts_dir, '*.ttf'))

    if not os.path.exists(label_file_2):
        os.makefiles(os.path.join(label_file_2))
        
    f = open(label_file_2, 'wt', encoding='utf-8') # create split no file
    
    for font in target_fonts:
    
        for i in labels_s_no:
            f.write(str(i))
    f.close()
    
    total_count = 0
    prev_count = 0
    font_count = 0
    char_no = 0
    
    for character in labels_s: # labels split imgs
        char_no += 1
        #if len(labels)
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))
            
        for x in range(len(target_fonts)):
            font_count += 1
            
            for font in fonts:
                total_count += 1
                image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
                font = ImageFont.truetype(font, 180)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                drawing.text(
                    ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                    character,
                    fill=(0),
                    font=font
                )
                file_string = '{:d}_{:05d}.jpeg'.format(font_count,char_no)
                file_path = os.path.join(image_dir, file_string)
                image.save(file_path, 'JPEG')
        font_count = 0
    char_no = 0

    print('Finished generating {} images.'.format(total_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--label-file-2', type=str, dest='label_file_2',
                        default=DEFAULT_LABEL_FILE_2,
                        help='File containing newline delimited labels.')
    parser.add_argument('--target-font-dir', type=str, dest='target_fonts_dir',
                        default=DEFAULT_TARGET_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--src-font-dir', type=str, dest='src_fonts_dir',
                        default=DEFAULT_SOURCE_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images.')
    args = parser.parse_args()

    generate_hangul_images(args.label_file, args.label_file_2,args.target_fonts_dir, args.src_fonts_dir, args.output_dir)
    