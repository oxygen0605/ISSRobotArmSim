"""One inference process and one shared controller stack for all tasks."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

def start(context):
    root=Path(get_package_prefix('iss_capture_ros')).parents[2]
    def value(k):return LaunchConfiguration(k).perform(context)
    args=['--task',value('task'),'--seed',value('seed'),'--checkpoint',value('checkpoint'),'--linger',value('linger')]
    if value('gui').lower()=='true':args+=['--gui']
    if value('record').lower()=='true':args+=['--record']
    return [Node(package='iss_capture_ros',executable='one_policy',name='biarm_one_policy',arguments=args,output='screen')]
def generate_launch_description():
    root=Path(get_package_prefix('iss_capture_ros')).parents[2]
    values=dict(task='task1',seed='3300000',checkpoint=str(root/'artifacts/smolvla_one_policy'),gui='true',record='false',linger='15')
    return LaunchDescription([DeclareLaunchArgument(k,default_value=v) for k,v in values.items()]+[OpaqueFunction(function=start)])
