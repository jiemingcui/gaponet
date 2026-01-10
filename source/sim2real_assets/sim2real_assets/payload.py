import numpy as np

from pxr import UsdPhysics, Gf

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.utils.stage import get_current_stage

reference_prim_path = "/World/h1_2/left_wrist_roll_link/payload_link"

translation = (0.05, 0.035, 0)
orientation = (1, 0, 0, 0)

prim_utils.create_prim(reference_prim_path, prim_type="Xform", translation=translation, orientation=orientation)


payload_prim_path = "/World/h1_2/payload"

prim = DynamicCylinder(
    prim_path=f"{payload_prim_path}/Cylinder",
    radius=0.02,
    height=0.06,
    color=np.array([1.0, 0.0, 0.0]),
    mass=1.0,
)

fixed_joint = UsdPhysics.FixedJoint.Define(get_current_stage(), f"{payload_prim_path}/FixedRootJoint")

fixed_joint.CreateBody0Rel().SetTargets([reference_prim_path])
fixed_joint.CreateBody1Rel().SetTargets([f"{payload_prim_path}/Cylinder"])

fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0.02, 0))

UsdPhysics.ArticulationRootAPI.Apply(fixed_joint.GetPrim())
