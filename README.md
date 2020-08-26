# Soccer-Ball-Tracking
Tracking of a Soccer ball has been done using a Random Forest Classifier. First the code splits the video into frames and then extracts features from it in a region of interest
. The region of interst ensures that noisy parts of the frames are kept out of analysis and only frames which have the ball are selected.
A trainning data set ( 110 Football Frames)
is used to classify features as ball or not.
On a frame ( which is an Image), once a feature has been classified as a ball a rectangle is drawn on the frame. 
Input file : soccer_02.mp4
Finally frames are combined together and embedded in the original video.
Executing the Code
1.Run Save_frames ( places each frame in a frames directory)
2.Run read_frames  ( detects ball, creates a scene selection chart and image of features and creates the final video)

