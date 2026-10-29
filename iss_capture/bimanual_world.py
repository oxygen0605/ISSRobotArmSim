"""Two official SO-101 arms, physical hollow cap, zero-gravity free pen."""
import copy,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .config import ROOT,JOINTS
from .world import convert_urdf,make_world,child,box,plugin,numbers,material
from .prop_meshes import compile_cabin_meshes
from .pen_kinematics import FIXED_PAD,FIXED_SIZE,MOVING_PAD,MOVING_SIZE
from .pen_world import friction
from .bimanual_config import *
from .bimanual_kinematics import BimanualArm

def look_at(eye,target):
    d=np.asarray(target)-eye
    return [*eye,0,float(np.arctan2(-d[2],np.linalg.norm(d[:2]))),float(np.arctan2(d[1],d[0]))]

def case_from_seed(seed,spread=.006):
    rng=np.random.default_rng(seed)
    return dict(seed=int(seed),pen_position=(np.array([.345,0.,.225])+rng.uniform(-spread,spread,3)).tolist(),
        cap_position=(np.array([.355,0.,.345])+rng.uniform(-spread,spread,3)).tolist(),
        pen_tilt=rng.uniform(-.025,.025,2).tolist(),cap_tilt=rng.uniform(-.025,.025,2).tolist(),
        linear_velocity=[0.,0.,0.],angular_velocity=[0.,0.,0.])

def rename_sdf(model,side):
    mapping={e.get('name'):side+'_'+e.get('name') for e in list(model) if e.tag in ('link','joint','frame')}
    for e in model.iter():
        for a in ('name','relative_to','attached_to','expressed_in'):
            if e.get(a) in mapping:e.set(a,mapping[e.get(a)])
        if e.tag in ('parent','child') and e.text in mapping:e.text=mapping[e.text]
    return model

def add_pads(robot,side,profile="flat"):
    for which,linkname in [('fixed','gripper_link'),('moving','moving_jaw_so101_v1_link')]:
        link=robot.find("link[@name='"+side+'_'+linkname+"']")
        for item in list(link):
            if item.tag in ('collision','sensor') or item.get('name')=='sponge_visual':link.remove(item)
        if which=='fixed':
            box(link,'fixed_pad',[.004,.030,.044],[*FIXED_PAD,0,0,0],'.06 .07 .08 1',True)
            for name,z in [('front',-.118),('back',-.066)]:
                box(link,'fixed_lip_'+name,[.009,.030,.008],[-.0064,-.0002,z,0,0,0],'.12 .13 .14 1',True)
            box(link,'palm',[.044,.038,.030],[.006,0,-.024,0,0,0],'.08 .09 .10 1',True)
        else:box(link,'moving_pad',MOVING_SIZE,[*MOVING_PAD,0,0,0],'.06 .07 .08 1',True)
        if profile=='v_groove' and which=='fixed':
            for item in list(link):
                if item.get('name','').startswith(('fixed_pad','fixed_lip')):link.remove(item)
            apex=-.012;centre_z=float(FIXED_PAD[2]);thickness=.003;span=.016
            box(link,'fixed_backing',[.006,.034,.040],[-.017,-.0002,centre_z,0,0,0],'.07 .08 .10 1',True)
            for direction in (-1,1):
                angle=direction*np.pi/4
                normal=np.array([np.cos(angle),0,-np.sin(angle)])
                surface=np.array([apex+span/2,-.0002,centre_z+direction*span/2])
                centre=surface-normal*thickness/2
                box(link,'fixed_v_'+str(direction),[thickness,.030,span*np.sqrt(2)+.001],
                    [*centre,0,angle,0],'.12 .14 .17 1',True)
        for c in link.findall('collision'):
            friction(c)
            if profile=='v_groove':
                mu=.25 if which=='fixed' else .8
                c.find('surface/friction/ode/mu').text=str(mu)
                c.find('surface/friction/ode/mu2').text=str(mu)
        sensor=child(link,'sensor',name=side+'_'+which+'_contact',type='contact')
        child(sensor,'always_on','true');child(sensor,'update_rate',200)
        contact=child(sensor,'contact')
        for c in link.findall('collision'):child(contact,'collision',c.get('name'))
        child(contact,'topic','/bimanual/'+side+'/'+which+'_contacts')

def add_control(robot,source_urdf,out,home):
    urdf=ET.Element('robot',name='bimanual_so101')
    child(urdf,'link',name='world')
    original=ET.parse(source_urdf).getroot()
    for side in ('left','right'):
        names={e.get('name'):side+'_'+e.get('name') for e in list(original) if e.get('name')}
        for old in original:
            item=copy.deepcopy(old)
            for e in item.iter():
                for key in ('name','link','joint'):
                    if e.get(key) in names:e.set(key,names[e.get(key)])
            urdf.append(item)
        fixed=child(urdf,'joint',name=side+'_mount',type='fixed')
        child(fixed,'parent',link='world');child(fixed,'child',link=side+'_base_link')
        child(fixed,'origin',xyz=numbers(BASE_POSES[side][:3]),rpy=numbers(BASE_POSES[side][3:]))
    control=child(urdf,'ros2_control',name='BimanualSystem',type='system')
    child(child(control,'hardware'),'plugin','gz_ros2_control/GazeboSimSystem')
    for i,name in enumerate(JOINT_NAMES):
        limit=urdf.find("joint[@name='"+name+"']/limit")
        limit.set('velocity',str(SPEEDS[i]))
        effort=i in GRIPPER_INDICES
        if effort:limit.set('effort','.01')
        j=child(control,'joint',name=name)
        cmd=child(j,'command_interface',name='effort' if effort else 'position')
        child(cmd,'param',-.01 if effort else limit.get('lower'),name='min')
        child(cmd,'param',.01 if effort else limit.get('upper'),name='max')
        pos=child(j,'state_interface',name='position');child(pos,'param',home[i],name='initial_value')
        child(j,'state_interface',name='velocity');child(j,'state_interface',name='effort')
    desc=out/'bimanual_control.urdf'
    ET.ElementTree(urdf).write(desc,encoding='utf-8',xml_declaration=True)
    params={NAMESPACE+'/controller_manager':{'ros__parameters':dict(update_rate=1000,enforce_command_limits=True,
             **{c:{'type':'joint_state_broadcaster/JointStateBroadcaster' if c==CONTROLLERS[0] else
                       'joint_trajectory_controller/JointTrajectoryController'} for c in CONTROLLERS})},
            NAMESPACE+'/joint_state_broadcaster':{'ros__parameters':dict(update_rate=200,joints=list(JOINT_NAMES),interfaces=['position','velocity','effort'])}}
    for name,indices,interface in [('arms_controller',ARM_INDICES,'position'),('grippers_controller',GRIPPER_INDICES,'effort')]:
        spec=dict(joints=[JOINT_NAMES[i] for i in indices],command_interfaces=[interface],state_interfaces=['position','velocity'],
                  interpolation_method='splines',interpolate_from_desired_state=True)
        if interface=='effort':
            spec['gains']={JOINT_NAMES[i]:dict(p=.08,i=0.,d=.008,ff_velocity_scale=0.,u_clamp_min=-.01,u_clamp_max=.01) for i in indices}
        params[NAMESPACE+'/'+name]={'ros__parameters':spec}
    config=out/'controllers.yaml';config.write_text(yaml.safe_dump(params,sort_keys=False))
    p=plugin(robot,'gz_ros2_control-system','gz_ros2_control::GazeboSimROS2ControlPlugin',
             parameters=str(config),position_proportional_gain='.008')
    ros=child(p,'ros');child(ros,'namespace',NAMESPACE)
    child(ros,'remapping','robot_description:='+NAMESPACE+'/robot_description')
    child(ros,'remapping','/joint_states:='+NAMESPACE+'/joint_states')
    return dict(robot_description=str(desc),parameters=str(config),joint_names=list(JOINT_NAMES),
        controllers=list(CONTROLLERS),trajectory_topics=list(TOPICS),velocity_limits=SPEEDS.tolist())

def add_camera(world,name,eye,target,size=(256,256),fov=.9):
    m=child(world,'model',name='capping_camera_'+name);child(m,'static','true')
    child(m,'pose',numbers(look_at(np.array(eye),np.array(target))))
    sensor=child(child(m,'link',name='body'),'sensor',name=name,type='camera')
    child(sensor,'always_on','true');child(sensor,'update_rate',10 if name=='display' else 20)
    child(sensor,'topic','/bimanual/display/image' if name=='display' else '/camera/'+name+'/image')
    cam=child(sensor,'camera');child(cam,'horizontal_fov',fov)
    im=child(cam,'image');child(im,'width',size[0]);child(im,'height',size[1]);child(im,'format','R8G8B8')
    clip=child(cam,'clip');child(clip,'near',.01);child(clip,'far',8)

def cylinder(link,name,radius,length,z,color=None):
    item=child(link,'visual' if color else 'collision',name=name)
    child(item,'pose',numbers([0,0,z,0,0,0]))
    geom=child(child(item,'geometry'),'cylinder');child(geom,'radius',radius);child(geom,'length',length)
    if color:material(item,color)
    else:friction(item)
    return item

def hollow_mesh(path):
    n=64;v=[]
    for z,r in [(-CAP_LENGTH/2,CAP_OUTER_RADIUS),(CAP_LENGTH/2,CAP_OUTER_RADIUS),
                (-CAP_LENGTH/2,CAP_INNER_RADIUS),(CAP_LENGTH/2-.003,CAP_INNER_RADIUS)]:
        v.extend([(r*np.cos(a),r*np.sin(a),z) for a in np.arange(n)*2*np.pi/n])
    faces=[]
    for a,b,reverse in [(0,n,False),(2*n,3*n,True),(0,2*n,True),(n,3*n,False)]:
        for i in range(n):
            j=(i+1)%n;quad=[a+i,a+j,b+j,b+i]
            if reverse:quad=quad[::-1]
            faces.extend([(quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3])])
    path.with_suffix('.mtl').write_text('newmtl cap_blue\nKa 0.06 0.30 0.85\nKd 0.06 0.30 0.85\nKs 0.25 0.25 0.25\nNs 60\n')
    lines=['# Original procedural hollow cap; metres.','mtllib '+path.with_suffix('.mtl').name,'usemtl cap_blue']+['v '+' '.join(map(str,p)) for p in v]
    lines+=['f '+' '.join(str(i+1) for i in f) for f in faces]
    path.write_text('\n'.join(lines)+'\n')

def build_capping(output,case=None,record_view=False,gripper_profile="flat"):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);case=case or case_from_seed(0)
    original=convert_urdf(out)
    for key in ('linear_velocity','angular_velocity'):
        value=np.asarray(case.get(key,[0,0,0]),dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Invalid initial motion: '+key)
    root=make_world(original,[.35,0,.3],appearance='training',environment='iss',view='wide')
    world=root.find('world')
    for name in ('so101','droplet'):world.remove(world.find("model[@name='"+name+"']"))
    pen=world.find("model[@name='floating_pen']")
    for p in pen.findall('plugin'):pen.remove(p)
    # The capped pen has a closed rear end, not an operable click button.
    pen.find("link/visual[@name='push_button']").set('name','rear_end')
    pen.find('pose').text=numbers([*case['pen_position'],np.pi+case['pen_tilt'][0],case['pen_tilt'][1],0])
    body=pen.find('link')
    cylinder(body,'pen_collision',.0054,.142,-.001)
    # Slender physical tip matches the visible nib and can enter the hollow cap.
    cylinder(body,'pen_tip_collision',.001,.009,-.0715)
    cap=child(world,'model',name='floating_cap')
    child(cap,'pose',numbers([*case['cap_position'],*case['cap_tilt'],0]))
    link=child(cap,'link',name='cap_body');child(link,'gravity','false')
    inertial=child(link,'inertial');child(inertial,'mass',.006)
    inertia=child(inertial,'inertia')
    for k,v in dict(ixx=1.6e-6,iyy=1.6e-6,izz=4e-7,ixy=0,ixz=0,iyz=0).items():child(inertia,k,v)
    mesh=out/'cap.obj';hollow_mesh(mesh)
    vis=child(link,'visual',name='blue_hollow_cap')
    child(child(child(vis,'geometry'),'mesh'),'uri',mesh.as_uri());material(vis,'.06 .30 .85 1')
    link.remove(vis)  # Primitive surfaces preserve the intended blue in Ogre2.
    cylinder(link,'cap_end_visual',CAP_OUTER_RADIUS,.003,CAP_LENGTH/2-.0015,'.06 .30 .85 1')
    cylinder(link,'cap_end_collision',CAP_OUTER_RADIUS,.003,CAP_LENGTH/2-.0015)
    # Convex wall segments leave the cavity open; no solid-cylinder shortcut.
    for i in range(32):
        angle=2*np.pi*i/32;r=(CAP_INNER_RADIUS+CAP_OUTER_RADIUS)/2
        c=child(link,'collision',name='cap_wall_'+str(i))
        child(c,'pose',numbers([r*np.cos(angle),r*np.sin(angle),0,0,0,angle]))
        child(child(child(c,'geometry'),'box'),'size',numbers([CAP_OUTER_RADIUS-CAP_INNER_RADIUS,2*CAP_OUTER_RADIUS*np.tan(np.pi/32),CAP_LENGTH]))
        friction(c)
        box(link,'cap_shell_'+str(i),[CAP_OUTER_RADIUS-CAP_INNER_RADIUS,2*CAP_OUTER_RADIUS*np.tan(np.pi/32),CAP_LENGTH],
            [r*np.cos(angle),r*np.sin(angle),0,0,0,angle],'.06 .30 .85 1')
    box(link,'cap_clip',[.002,.003,.032],[CAP_OUTER_RADIUS+.001,0,.004,0,0,0],'.70 .75 .80 1')
    for model in (pen,cap):
        plugin(model,'gz-sim-pose-publisher-system','gz::sim::systems::PosePublisher',
               publish_model_pose='true',publish_link_pose='false',use_pose_vector_msg='true',
               update_frequency=100,topic='/evaluation/object_pose')
    robot=child(world,'model',name='bimanual_so101')
    child(robot,'self_collide','true')
    home=[]
    for side,pos in [('left',np.array([.245,.11,.225])),('right',np.array([.245,-.11,.345]))]:
        # Fixed, case-independent home avoids exposing object state in proprioception.
        home.extend(BimanualArm(side).initial(pos).tolist())
        arm=rename_sdf(copy.deepcopy(original),side)
        base=arm.find("link[@name='"+side+"_base_link']")
        child(base,'pose',numbers(BASE_POSES[side]))
        for item in list(arm):robot.append(item)
        fixed=child(robot,'joint',name=side+'_world_fixed',type='fixed')
        child(fixed,'parent','world');child(fixed,'child',side+'_base_link')
        add_pads(robot,side,gripper_profile)
    mount=child(world,'model',name='right_mount');child(mount,'static','true')
    box(child(mount,'link',name='body'),'support',[.12,.12,.038],[0,-.18,.006,0,0,0],'.20 .23 .28 1')
    mount_left=child(world,'model',name='left_mount');child(mount_left,'static','true')
    box(child(mount_left,'link',name='body'),'support',[.12,.12,.038],[0,.18,.006,0,0,0],'.20 .23 .28 1')
    control=add_control(robot,out/'so101_resolved.urdf',out,np.array(home))
    plugin(world,'gz-sim-sensors-system','gz::sim::systems::Sensors',render_engine='ogre2')
    plugin(world,'gz-sim-apply-link-wrench-system','gz::sim::systems::ApplyLinkWrench')
    add_camera(world,'front',[-.08,0,.48],[.31,0,.28],fov=1.25)
    add_camera(world,'side',[.37,-.50,.39],[.31,0,.28],fov=.95)
    eye=[-.40,-.76,.72];target=[.24,0,.22]
    world.find("gui/plugin[@filename='MinimalScene']/camera_pose").text=numbers(look_at(np.array(eye),np.array(target)))
    world.find("gui/plugin[@filename='MinimalScene']/horizontal_fov").text='55'
    world.find("gui/plugin[@filename='MinimalScene']/gz-gui/title").text='ISS / Side-by-side SO-101 / One policy'
    if record_view:add_camera(world,'display',eye,target,(960,640),fov=np.deg2rad(55))
    # Extend the existing cantilever work surface under both mounted arms.
    deck=world.find("model[@name='station_panel']/link/visual[@name='deck_visual']")
    deck.find('pose').text='.35 0 -.025 0 0 0';deck.find('geometry/box/size').text='.90 .75 .025'
    compile_cabin_meshes(world,out/'cabin_meshes')
    ET.indent(root,space='  ');path=out/'bimanual_capping.sdf'
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    check=subprocess.run(['gz','sdf','-k',str(path)],capture_output=True,text=True,timeout=60)
    (out/'validation.log').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError(check.stdout+check.stderr)
    manifest=dict(task='bimanual_pen_capping',version=3 if gripper_profile=='v_groove' else 2,case=case,home_q=home,joint_control=control,
                  gravity=[0,0,0],objects_free=True,grasp='frictional contacts; no attachment',gripper_profile=gripper_profile,
                  cap_cavity=True,policy_input=['RGB front','RGB side','previous front','12 joint positions','text'],
                  success='both objects grasped, pen tip inside aligned cap at 29-38 mm depth and low relative speed for 0.6 seconds',
                  cap_retention='geometric fit while held; no elastic snap-fit or artificial fixed joint')
    (out/'scene.json').write_text(json.dumps(manifest,indent=2))
    return path,manifest
