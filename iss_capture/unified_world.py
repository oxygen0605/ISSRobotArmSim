"""One fixed robot/tool/camera scene shared by all three tasks."""
import json, xml.etree.ElementTree as ET
from pathlib import Path
from .bimanual_world import build_capping
from .world import child,box,plugin,numbers,material
from .appearance import apply_droplet_appearance
from .water_plugin import install_water_plugins
from .unified_config import SPONGE,SPONGE_SIZE,CAMERAS
from .config import DROPLET_RADIUS
import numpy as np

def build(output,case,record=False):
    path,manifest=build_capping(output,case,record,'v_groove')
    root=ET.parse(path).getroot();world=root.find('world')
    link=world.find("model[@name='bimanual_so101']/link[@name='left_gripper_link']")
    box(link,'sponge',SPONGE_SIZE,[*SPONGE,0,0,0],'.45 .95 .58 1',True)
    sensor=child(link,'sensor',name='sponge_touch',type='contact')
    child(sensor,'always_on','true');child(sensor,'update_rate',1000)
    contact=child(sensor,'contact');child(contact,'collision','sponge_collision');child(contact,'topic','/capture/contacts')
    drop=child(world,'model',name='droplet');child(drop,'static','false')
    child(drop,'pose',numbers([*case['water_position'],0,0,0]))
    body=child(drop,'link',name='body')
    mass=1000*4*np.pi*DROPLET_RADIUS**3/3
    inertial=child(body,'inertial');child(inertial,'mass',mass);tensor=child(inertial,'inertia')
    for tag in ('ixx','iyy','izz'):child(tensor,tag,.4*mass*DROPLET_RADIUS**2)
    for tag in ('ixy','ixz','iyz'):child(tensor,tag,0)
    for kind,name in [('collision','water_collision'),('visual','water_visual')]:
        item=child(body,kind,name=name);child(child(child(item,'geometry'),'sphere'),'radius',DROPLET_RADIUS)
        if kind=='visual':material(item,'.05 .65 1 1')
    depth_visual=child(body,'visual',name='water_depth_sphere')
    child(depth_visual,'visibility_flags',1<<24);child(depth_visual,'cast_shadows','false')
    child(child(child(depth_visual,'geometry'),'sphere'),'radius',DROPLET_RADIUS)
    material(depth_visual,'.3 .3 .3 1')
    apply_droplet_appearance(world,'refractive')
    optics=install_water_plugins(root,.06)
    server=world.find("plugin[@name='iss::WaterSystem']")
    server.find('sponge_center').text=numbers(SPONGE)
    child(server,'robot_model','bimanual_so101');child(server,'sponge_link','left_gripper_link')
    plugin(drop,'gz-sim-pose-publisher-system','gz::sim::systems::PosePublisher',
        publish_model_pose='true',publish_link_pose='false',use_pose_vector_msg='true',update_frequency=100,topic='/evaluation/object_pose')
    # Co-located RGB and depth sensors provide observable 3D geometry.
    for name in ('front','side'):
        camera_link=world.find("model[@name='capping_camera_"+name+"']/link")
        camera_link.find('sensor/update_rate').text='10'
        rgb=camera_link.find('sensor/camera')
        rgb.find('image/width').text='512';rgb.find('image/height').text='512'
        child(rgb,'triggered','true');child(rgb,'trigger_topic','/camera/capture')
        sensor=child(camera_link,'sensor',name=name+'_depth',type='depth_camera')
        child(sensor,'always_on','true');child(sensor,'update_rate',10);child(sensor,'topic','/camera/'+name+'/depth')
        camera=child(sensor,'camera');child(camera,'horizontal_fov',CAMERAS[name]['fov'])
        child(camera,'triggered','true');child(camera,'trigger_topic','/camera/capture')
        im=child(camera,'image');child(im,'width',512);child(im,'height',512);child(im,'format','R_FLOAT32')
        clip=child(camera,'clip');child(clip,'near',.01);child(clip,'far',8)
    for camera in world.findall(".//sensor[@type='camera']/camera"):child(camera,'visibility_mask',0xFFFFFFFF^(1<<24))
    ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    manifest.update(scene_version=4,water_depth_model='idealized opaque depth-only sphere, excluded from RGB and refraction; follows visible water adhesion',task='shared_three_task_scene',camera_calibration=CAMERAS,water_optics=optics,
        sponge=dict(link='left_gripper_link',center=SPONGE.tolist(),size=SPONGE_SIZE),
        layout='both bases on x=0 table edge, y=+/-0.18 m, z=0.025 m, both facing +X',
        policy_input=['egocentric RGB-D','side RGB-D','previous egocentric RGB','encoders and joint velocities','language'],
        all_objects_present_for_every_task=True,camera_resolution=[512,512],depth_topics=['/camera/front/depth','/camera/side/depth'],camera_trigger_topic='/camera/capture',camera_capture='synchronized request while physics is paused')
    (Path(output)/'scene.json').write_text(json.dumps(manifest,indent=2))
    return path,manifest
