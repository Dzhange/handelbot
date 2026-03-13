from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_closest_joint_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    target_qpos: onp.ndarray,
    curr_qpos: onp.ndarray,
    world_coll_list: Sequence[pk.collision.CollGeom] | None = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_qpos: onp.ndarray. Shape: (robot.joint.actuated_count,). Target qpos.
        curr_qpos: onp.ndarray. Shape: (robot.joint.actuated_count,). Current qpos.
        world_coll_list: Sequence[pk.collision.CollGeom]. List of world collision objects.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    if world_coll_list is None:
        world_coll_list = []
    
    is_batched = target_qpos.ndim == 2
    if is_batched:
        assert target_qpos.shape[1] == robot.joints.num_actuated_joints
        assert curr_qpos.shape[1] == robot.joints.num_actuated_joints
        
        solve_batch = jax.vmap(
            _solve_closest_joint_with_collision_jax_pos, 
            in_axes=(None, None, 0, 0, None)
        )
        
        cfg = solve_batch(
            robot,
            coll,
            target_qpos,
            curr_qpos,
            world_coll_list,
        )
        
    else:
        assert target_qpos.shape == (robot.joints.num_actuated_joints,), f"{target_qpos.shape} != {robot.joints.num_actuated_joints}"
        assert curr_qpos.shape == (robot.joints.num_actuated_joints,), f"{curr_qpos.shape} != {robot.joints.num_actuated_joints}"

        cfg = _solve_closest_joint_with_collision_jax_pos(
            robot,
            coll,
            target_qpos,
            curr_qpos,
            world_coll_list,
        )
        assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_closest_joint_with_collision_jax_pos(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    target_qpos: jax.Array,
    curr_qpos: jax.Array,
    world_coll_list: Sequence[pk.collision.CollGeom],
) -> jax.Array:
    JointVar = robot.joint_var_cls

    costs = [
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=target_qpos,
            weight=7.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=JointVar(0),
            margin=0.02,
            weight=45.0,
        ),
    ]
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, coll, JointVar(0), world_coll, 0.005, 100.0
            )
            for world_coll in world_coll_list
        ]
    )
    sol = (
        jaxls.LeastSquaresProblem(costs, [JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
        )
    )
    return sol[JointVar(0)]