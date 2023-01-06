# Direction Prediction Refinement
In our post-process, motion/direction information is very important, accurate direction prediction can greatly improve our final performance, to test the upper bound of our model, so we propose some methods to do direction refinement of the visual tracks in test set.

This module will do the following tasks:
- Refine the test direction prediction for the tracks in the shared test cameras.
- Refine the test direction prediction for the tracks in the unique test cameras.

Specifically, we argue that the movement direction of the vehicle is mostly decided by the road, cause there is some specifically traffic rules that they must obey, so we draw the road mask of each camera, and predict the vehicle's direction by their position on the road mask.
## Module organization 
- `data`: stores some necessary data files which is used to do the direction refinement, which includes 'imgs': road mask info under the shared test cameras, 'masks: road mask images, 'test_det.json','track2.89.bbox.json': vehicle detection result which is used in the stop_detector. These files can download in (https://pan.baidu.com/s/1j1usbiJQoxdmdedNDApzaQ?pwd=2pgh code: 2pgh)
- `road_mask`: the road mask we annotated for the shared cameras in test set. 
- `unique_road_mask`: the road mask we annotated for unique cameras in test set.
- `results`: stores the final refined test direction prediction results.

## Test direction refinement
You need to put the direction classifier's prediction result into the data folder, that is 'target_test_direction_prediction_one_hot.json', after running following command, you will get the refined direction prediction, that is 'target_test_direction_prediction_one_hot_refinement.json'.
```
python utils.py             # generate test_info.json
python get_direction.py     # get the direction refinement results
```

Please pay attention that, even though we annotate the road mask of the test tracks, we didn't use these direction prediction results to get our final submission, and just to test the upper bound of our model.