Step 1

subst M: "C:\\Users\\j08490dd\\OneDrive - The University of Manchester\\Documents\\00-unsorted\\training\\Multiverse"

\--------------

Step 2

Generate the source the AVI file in legacy software

\------

Step 3



Download the ffmpeg package https://github.com/GyanD/codexffmpeg/releases/tag/2026-07-16-git-ceabc9b306

\-----

Step 4: Create or pull GitHub repo

AIML\_Fellowship



\-----------

Step 5 

Create GitHub project and Kanban Board for online collaboration 

\------------

Step 6

Create a Kanban issue for PowerShell script 01\_get\_avi\_hash.ps1 to get the hash of an AVI file 

and push to GitHub

\--------

Step 7

**Create a Kanban issue for PowerShell script 02\_get\_metadata\_ffprob.ps1 to get metadata of an AVI file** 

**and push to GitHub**

\----------

Step 8

**Create a Kanban issue for PowerShell script 03\_get\_avi\_metadata\_powershell.ps1 to get metadata of an AVI file**

**and push to GitHub**

\---------------

Step 9



Create a PowerShell script to extract frames from a video and save them in the Frame -> Origin folder

04\_extract\_frames from\_avi.ps1

and push to GitHub

\--------------

Step 10



Create a PowerShell script to extract the ROI from all frames - Region of Interest (ROI) or crop rectangle

ffmpeg -i try.avi -vf "crop=430:570:9:109" frame\_%04d.png

and push to Git Hub



\------



Step 11



Install python3 and missing modules

py -m pip install opencv-python numpy

\-----------







