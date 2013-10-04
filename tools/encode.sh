#!/bin/bash

#================================================================#
#                                                                #
# Script to encode .avi movies playable on windows machines      #
# Requires arguments :                                           #
# 1st - image file type (e.g. jpg,png,etc...)                    #
# 2nd - path to folder containing images, and for movie output.  #
# 3rd - name of movie output file (without the file extension).  #
# 4th - OPTIONAL - enter characters here to limit the image      #
#                 files used as frames (e.g. 'image1' would only #
#                 select images which satisfy 'image1*.jpg').    #
#                                                                #
# e.g. "encode.sh jpg /data/movie_files my_movie image002"       #
# will encode a movie called my_movie using all jpg files        #
# from the directory /data/movie_files/ starting with the        #
# characters 'image002'                                          #
#                                                                #
# Feel free to change fps (frames per second) and BITRATE        #
# For optimum value work bitrate out as:                         #
#                                                                #
# BITRATE = 1250*<no of pixels in x>*<no of pixels in y>/256     #
#                                                                #
#================================================================#

FILETYPE=$1
DIRNAME=$2
OUTPUTNAME=$3
IMAGEDIRNAME=$DIRNAME

# mencoder does not like gif images...
if [ $FILETYPE = "gif" ]; then
   # Make the directory
   PNGDIRNAME=$DIRNAME"/encode-png"
   mkdir "-p" $PNGDIRNAME
   echo "============================================================= "
   echo "! mencoder does not like gif files, so creating a directory ! "
   echo "! with png images to use to make the movie. This may take   ! "
   echo "! some time. The png images are located in:                 ! "
   echo $PNGDIRNAME
   echo "============================================================= "
   echo
   # Convert the images
   for file in $DIRNAME/*.gif; do convert "$file" $PNGDIRNAME"/$(basename $file .gif).png"; done
   FILETYPE="png"
   IMAGEDIRNAME=$PNGDIRNAME
fi

# The input line for mencoder. change fps if you want a slower/faster movie
JOPT="mf://$IMAGEDIRNAME/"$4"*."$FILETYPE" -mf type="$FILETYPE":fps=25"
echo $JOPT

# Bitrate (this is worked out for ~600x800 image, change if your image is much different in size)
BITRATE=2160000

# Only change this if you know what you are doing....!
VOPT="vbitrate=$BITRATE:mbd=2:keyint=132:vqblur=1.0:cmp=2:subcmp=2:dia=2:mv0:last_pred=3"
echo $VOPT

# The important bit.
mencoder $JOPT -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=1:$VOPT -nosound -o /dev/null
mencoder $JOPT -ovc lavc -lavcopts vcodec=msmpeg4v2:vpass=2:$VOPT -nosound -o $DIRNAME/$OUTPUTNAME.avi

# Remove the log file
rm -rf divx2pass.log

# Check we made a movie
if [ -f $DIRNAME/$OUTPUTNAME.avi ]; then
   echo
   echo "=============================="
   if [ $1 = "gif" ]; then
      echo "# png images created in:     #"
      echo $PNGDIRNAME
      echo "=============================="
   fi
   echo "# All done, enjoy the movie! #"
   echo "=============================="
   echo
else
   echo
   echo "=================================================================="
   echo "! No .avi file produced. Check inputs are correct and try again. !"
   echo "!                                                                !"
   echo "! Input arguments should be:                                     !"
   echo "!  1st - image file type (e.g. jpg,png - not gif!)               !"
   echo "!  2nd - path to folder containing images, and for movie output. !"
   echo "!  3rd - name of movie output file (without the file extension). !"
   echo "!  4th - OPTIONAL - enter characters here to limit the image     !"
   echo "!                 files used as frames (e.g. 'image1' would only !"
   echo "!                 select images which satisfy 'image1*.jpg').    !"
   echo "!                                                                !"
   echo "!  e.g. 'encode.sh jpg /data/movie_files my_movie image002'      !"
   echo "!  will encode a movie called my_movie using all jpg files       !"
   echo "!  from the directory /data/movie_files/ starting with the       !"
   echo "!  characters 'image002'                                         !"
   echo "=================================================================="
fi

# All done.
