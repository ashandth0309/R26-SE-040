cmake_minimum_required(VERSION 3.8)

project(buddy_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}

  "msg/RobotStatus.msg"
  "msg/MovementCommand.msg"
  "msg/ObstacleStatus.msg"
  "msg/SafetyState.msg"
  "msg/RecognizedPerson.msg"
  "msg/EmotionObservation.msg"
  "msg/VoiceCommand.msg"
  "msg/SecurityEvent.msg"
  "msg/VisionObject.msg"
  "msg/VisionObjects.msg"
  "msg/VisionPerson.msg"
  "msg/VisionPeople.msg"
  "msg/VisionEvent.msg"
  "msg/SpeechRequest.msg"

  "srv/SetRobotMode.srv"
  "srv/EmergencyStop.srv"
  "srv/ResetEmergencyStop.srv"
  "srv/VisionCommand.srv"
  "srv/IdentifyObject.srv"
  "srv/TeachObject.srv"
  "srv/WhereIs.srv"
  "srv/RouteUtterance.srv"

  "action/FollowPerson.action"
  "action/ApproachCaller.action"
  "action/PatrolRoute.action"

  DEPENDENCIES
  builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
