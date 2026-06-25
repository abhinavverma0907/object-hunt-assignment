# The Great Object Hunt ? Session 4

A ROS 2 + Gazebo robot that uses a camera and YOLOv8 to find an object,
walk up to it, and stop.

Pipeline: **SEARCH ? DETECT ? TRACK ? APPROACH ? COMPLETE**

## What's inside

- `erc_gazebo_sensors/` ? the robot, with a camera + depth sensor added
- `object_hunt/` ? the node that does the searching, tracking, and driving

## What it does

1. You type an object name (e.g. `fire hydrant`) in the terminal.
2. If the robot can't see it, it spins in place looking for it.
3. Once it sees the object, it measures the distance using the depth camera.
4. It turns to face the object, then drives toward it.
5. When it gets close enough, it stops and says "Mission Completed."
6. It then asks for the next target ? no need to restart.

## How to run it

**Terminal 1 ? start the simulation:**
```bash
ros2 launch erc_gazebo_sensors spawn_robot.launch.py world:=empty.sdf
```

**Terminal 2 ? spawn an object** (use one YOLO recognizes, like `Fire Hydrant`):
```bash
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'sdf_filename: "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Fire Hydrant", pose: {position: {x: 2.0, y: 0, z: 0}}'
```

**Terminal 3 ? run the hunter:**
```bash
ros2 run object_hunt object_hunter
```

Then type the object's name when asked.

## Notes

- The distance is measured by reading a small patch of the depth camera
  around the object, not just one pixel, to avoid noisy readings.
- YOLOv8 sometimes mislabels simulated objects since it was trained on real
  photos ? this doesn't affect how the robot searches, tracks, or stops.

Full explanation is in `Object_Hunt_Short_Report.docx`.
