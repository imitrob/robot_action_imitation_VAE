Data format:

poses.pkl is the list of extracted poses for given action. It is a list of numpy arrays, where each array is the pose at given timestep. It is a vector with 8 values:

[Left Shoulder Roll, Left Elbow Roll, Right Shoulder Roll, Right Elbow Roll, Left Arm Direction, Right Arm Direction, Left Hand Open, Right Hand Open]

The Roll values are the given joint angles in degrees.

Left Arm and Right Arm Direction 
- tells us basically the Shoulder Pitch, only it is not an angle (OpenPose does not allow extracting such angle from RGB images), but a binary value (it was retrieved based on the condition "is the wrist above the elbow", e.g. is the arm pointing upwards). In short, if it has value 100, the arm is pointing upwards, if it is 0, it is pointing downwards. 


Left Hand and Right Hand Open 
- again a binary value which tells whether the robot should open or close it's hand. 0 is for the closed hand, 100 is for an open hand. 
 
