#!/bin/bash
echo "Downloading OpenPose Body model..."
gdown '1EULkcH_hhSU28qVc1jSJpCh2hGOrzpjK'
mv body_pose_model.pth model/body_pose_model.pth
echo "done"

echo "Downloading OpenPose Hand model..."
gdown '1yVyIsOD32Mq28EHrVVlZbISDN7Icgaxw'

mv hand_pose_model.pth model/hand_pose_model.pth

echo "All done. Models are in the ./models/ folder."
