subst M: "C:\Users\j08490dd\OneDrive - The University of Manchester\Documents\00-unsorted\training\Multiverse"
--------------

Download the AVI file
------
Download the ffmpeg package https://github.com/GyanD/codexffmpeg/releases/tag/2026-07-16-git-ceabc9b306

AVI file Metadata
--------
01_get_avi_hash.ps1
--------
ffprobe -v quiet -print_format json -show_format -show_streams video.avi

----------
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 try.avi
.\bin\ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 .\origin\try.avi
302

---------------
mkdir frames

ffmpeg -i try.avi frames\frame_%04d.png
--------------
Extract the ROI from all frames - Region of Interest (ROI) or crop rectangle

ffmpeg -i try.avi -vf "crop=430:570:9:109" frame_%04d.png
------

