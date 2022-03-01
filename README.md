# PCO-Python-Processing

## Installation
1. Install python3.7.10
2. Install conda
3. Clone this repo onto your device and change directories into it
```sh
git clone https://github.com/Awallace3/PCO-Python-Processing.git && cd PCO-Python-Processing
```
4. Install the required dependencies from the environment.yml file to create a virtual environment
```sh
conda env create -f environment.yml
```
5. Open jupyter notebooks through your preferred method and select the "trackpy" kernel
6. Execute the first code block to ensure that everything worked properly
   (click in the box and press SHIFT+ENTER)

## Usage

1. Execute the first block of code to load all the libraries and functions (click in the box and press SHIFT+ENTER).

2. In the second block (with the following first line: v, t2, n, A, D =
   mainflow(...) , change video_path to be the path to the video you want to
   load into the script and execute the code block. Please rename the video
   file without whitespaces. e.g. change... video_path=r'old_video.mp4',
   size=7, memory=10, to... video_path=r'your_new_video.mp4' a. If this
   produces errors, then skip to step 4.

3. Execute "annotate_video(v, t2)" (code block 3 with SHIFT+ENTER) to see see
   which particles are being tracked. a. If only the particles you want with no
   other particles consistently appearing throughout the video, then you may
   take the diffusivity (D) from code block 2. Else, continue to step 4.

4.  If Step 2 produced errors or the output from block 2 under "t1:" has less
    than 5 frames, increase the size by increments of 2 with the size value
    being an odd number. An example is below e.g. change...
    video_path=r'your_new_video.mp4', size=5, memory=10, to....
    video_path=r'your_new_video.mp4', size=7, memory=10, a. if more frames do
    not appear under "t1:", then possibly invertColor or adjust the threshold
    for darkening the background. b. If these all fail, possibly retake the
    video with more time.  

5. If the output from block 2 under "t2:" is less than 5 frames, then analyze
   under "t1:" to see what min_mass, size, and ecc values your particle likely
   has. You can determine what values your particle is approximately by
   comparing the frame numbers with annotate_video(v, t2) and lining up which
   particle it is describing in each frame. a. Once you know what the
   approximate size mass and ecc your particle has, adjust the values in code
   block 2 to change the filter constraints. Not all the values need to be
   changed but enough filtering needs to occur to cut down on miscellaneous
   particles. e.g. change... min_mass=20, max_size=7, max_ecc=0.5, to...
   min_mass=30, max_size=5, max_ecc=0.3,

6. Continue step 5 until you are sure trackpy is analyzing mostly just your
   intended particles.

7. Take the diffusivity (D) from the bottom of the output of code block 2.

	
