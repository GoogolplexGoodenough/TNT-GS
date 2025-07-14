#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations
import os
import datetime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.render_TNT import render
from gaussian_renderer import network_gui
import sys
from scene import Scene
from scene.gaussian_model_TNT import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
import numpy as np
import wandb
from utils.image_utils import psnr , apply_dog_filter


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_to_wandb, output_size):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.prune_contrib_thres)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    init_point_size = gaussians._xyz.shape[0]
    target_points = init_point_size * 10
    if output_size <= 0:
        target_points = init_point_size * 10
    else:
        target_points = output_size * 4000

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    opt.densification_interval = len(scene.getTrainCameras()) + 1
    opt.opacity_reset_interval = round(opt.opacity_reset_interval / opt.densification_interval) * opt.densification_interval
    # res_per_loss = ResNetPerceptualLoss().to(gaussians._xyz.device)
    densi_times = int(np.floor((opt.densify_until_iter - opt.densify_from_iter) / opt.densification_interval)) + 1
    point_lists = [init_point_size + (target_points - init_point_size) / densi_times * (i + 1) for i in range(densi_times)]
    densify_iteration = 0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), dynamic_ncols=True, ascii=True)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        
        image = torch.nan_to_num(image, 0, 0)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0

        # rend_dist = render_pkg["rend_dist"]
        # dist_loss = lambda_dist * (rend_dist).mean()

        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
        else:
            normal_loss = 0.0

        if viewpoint_cam.gt_alpha_mask is not None:
            normal_loss = normal_loss + l1_loss(render_pkg['rend_alpha'] * (1 - viewpoint_cam.gt_alpha_mask), torch.zeros_like(render_pkg['rend_alpha']))
        total_loss = loss + normal_loss

        try:
            total_loss.backward()
        except Exception as e:
            print(e)

            import pdb
            pdb.set_trace()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # ema_per_loss_for_log = 0.4 * per_loss.item() + 0.6 * ema_per_loss_for_log
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            # ema_normal_loss_for_log = 0.4 * (tv_loss + normal_loss).item() + 0.6 * ema_normal_loss_for_log
            if lambda_normal > 0:
                ema_normal_loss_for_log = 0.4 * (normal_loss).item() + 0.6 * ema_normal_loss_for_log
    
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "normal_loss": f"{ema_normal_loss_for_log:.{7}f}",
                        "#Points": f"{gaussians._xyz.shape[0]}",
                        "out_size: M": f"{gaussians._xyz.shape[0] / 4000:.{2}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(log_to_wandb, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1, None))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, render_pkg["contrib_tensor"], visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    point_times = point_lists[densify_iteration]  / gaussians._xyz.shape[0]
                    densify_iteration += 1
                    # print(point_times, densi_times, np.exp(np.log(point_times) / densi_times) - 1)
                    if point_times > 1:
                        ratio = point_times - 1
                        
                        gaussians.densify_and_prune(
                            ratio, opt.prune_opacity_threshold, scene.cameras_extent, iteration > opt.opacity_reset_interval
                        )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # if (iteration + 1) % 5 == 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(log_to_wandb, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if log_to_wandb:
        wandb.log({
            'train_loss_patches/l1_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed,
            'scene/total_points': scene.gaussians.get_xyz.shape[0],
            'scene/opacity_grads':scene.gaussians._opacity.grad.data.norm(2).item(),
        }, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if log_to_wandb and (idx < 5):
                        wandb_key = f"renders/{config['name']}_view_{viewpoint.image_name}/render"
                        wandb.log({wandb_key: [wandb.Image(image, caption="Render at iteration {}".format(iteration))]}, step=iteration)
                        if iteration == testing_iterations[0]:
                            wandb_key = "renders/{}_view_{}/ground_truth".format(config['name'], viewpoint.image_name)
                            wandb.log({wandb_key: [wandb.Image(gt_image, caption="Ground truth at iteration {}".format(iteration))]}, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if log_to_wandb:
                    wandb.log({
                        f"metrics/{config['name']}/loss_viewpoint - l1_loss": l1_test,
                        f"metrics/{config['name']}/loss_viewpoint - psnr": psnr_test,
                    }, step=iteration)

        if log_to_wandb:
            opacity_data = [[val] for val in scene.gaussians.get_opacity.cpu().squeeze().tolist()]
            wandb.log({
                "scene/opacity_histogram": wandb.plot.histogram(wandb.Table(data=opacity_data, columns=["opacity"]), "opacity", title="Opacity Histogram"),
                "scene/total_points": scene.gaussians.get_xyz.shape[0],
            }, step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * i for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--nowandb", action="store_false", dest='wandb')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_size", type=int, default=50)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # random_id()
    # args.model_path = args.model_path + "_" + args.exp_set + "_" +  exp_id
    print("Optimizing " + args.model_path)
    safe_state(args.quiet, args.seed)
    setup = vars(args)
    setup["exp_id"] = exp_id
    if args.wandb:
        wandb_id = args.model_path.replace('outputs', '').lstrip('./').replace('/', '---')
        wandb.init(project="scan24", id=exp_id, config = setup ,sync_tensorboard=False,settings=wandb.Settings(_service_wait=600))

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.wandb, args.output_size)

    # All done
    print("\nTraining complete.")
    if args.wandb:
        wandb.finish()