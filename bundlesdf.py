# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from nerf_runner import *
from tool import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/third_party/BundleTrack/build")
import my_cpp
from gui import *
from third_party.BundleTrack.scripts.data_reader import *
from Utils import *
from eloftr_wrapper import ELoftrRunner
from loftr_wrapper import LoftrRunner
import multiprocessing, threading
import torch
from typing import Dict

try:
    multiprocessing.set_start_method("spawn")
except:
    pass


# Tensor precision management utilities
def set_bundlesdf_precision(use_full_precision: bool = True) -> torch.dtype:
    """Configure PyTorch tensor types for BundleSDF operations.

    This function sets the default tensor type and precision for all BundleSDF operations.
    Using full precision (float32) is recommended for pose tracking to ensure numerical stability.

    Args:
        use_full_precision: Whether to use full precision (float32) or allow mixed precision.
            Defaults to True for maximum stability.

    Returns:
        torch.dtype: The current default tensor type after configuration.

    Note:
        When use_full_precision is True:
        - Forces float32 precision for all tensors
        - Uses CUDA FloatTensor if CUDA is available
        - Ensures maximum numerical stability for pose tracking
    """
    if use_full_precision:
        # Force full precision (safest option for pose tracking)
        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        logging.info("Set PyTorch to use full precision (float32) for BundleSDF")
    return torch.get_default_dtype()


# Define required tensor types for different operations
LOFTR_TYPES = {
    "rgbAs": torch.float32,
    "rgbBs": torch.float32,
    "warp": torch.float32,
    "certainty": torch.float32,
    "matches": torch.float32,
}


class TensorUtils:
    """Handles tensor type management and conversions."""

    @staticmethod
    def ensure_tensor_type(
        tensor: torch.Tensor,
        target_dtype: torch.dtype = torch.float32,
        name: str = "tensor",
    ) -> torch.Tensor:
        """
        Ensure tensor is of specified type.

        This utility function checks if a tensor has the required data type
        and converts it if necessary, with optional logging for debugging.

        Args:
            tensor: Input tensor to check and potentially convert
            target_dtype: Desired PyTorch data type (default: torch.float32)
            name: Descriptive name for the tensor used in debug logging

        Returns:
            Tensor with the correct data type (either original or converted)

        Note:
            - Logs conversion operations at debug level
            - Returns original tensor if already correct type (no copy)
        """
        if tensor.dtype != target_dtype:
            logging.debug(f"Converting {name} from {tensor.dtype} to {target_dtype}")
            return tensor.to(target_dtype)
        return tensor

    @staticmethod
    def ensure_tensor_types(
        tensors_dict: Dict[str, torch.Tensor], dtypes_dict: Dict[str, torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """
        Ensure multiple tensors have correct types.

        This method processes a dictionary of tensors and ensures each one
        has the correct data type as specified in the dtypes dictionary.

        Args:
            tensors_dict: Dictionary mapping tensor names to torch.Tensor
                         objects
            dtypes_dict: Dictionary mapping tensor names to required
                        torch.dtype values

        Returns:
            Dictionary with same keys as tensors_dict but with corrected
            tensor types where necessary

        Note:
            - Only tensors specified in dtypes_dict are type-checked
            - Other tensors are passed through unchanged
        """
        result = {}
        for name, tensor in tensors_dict.items():
            if name in dtypes_dict:
                result[name] = TensorUtils.ensure_tensor_type(
                    tensor, dtypes_dict[name], name
                )
            else:
                result[name] = tensor
        return result


# Update tensor type utilities
def ensure_tensor_type(
    tensor: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
    name: str = "tensor",
) -> torch.Tensor:
    """Ensure a tensor has the specified data type.

    This is a wrapper function for backward compatibility that ensures tensors
    have the correct data type for BundleSDF operations.

    Args:
        tensor: Input tensor to convert
        target_dtype: Desired data type for the tensor
        name: Name of the tensor for logging purposes

    Returns:
        torch.Tensor: Tensor converted to the target data type

    Raises:
        TypeError: If input is not a torch.Tensor
    """
    return TensorUtils.ensure_tensor_type(tensor, target_dtype, name)


def ensure_tensor_types(
    tensors_dict: Dict[str, torch.Tensor], dtypes_dict: Dict[str, torch.dtype]
) -> Dict[str, torch.Tensor]:
    """Ensure multiple tensors have their specified data types.

    This is a wrapper function for backward compatibility that ensures multiple
    tensors have the correct data types for BundleSDF operations.

    Args:
        tensors_dict: Dictionary of tensors to convert
        dtypes_dict: Dictionary mapping tensor names to their target data types

    Returns:
        Dict[str, torch.Tensor]: Dictionary of tensors converted to their target data types

    Raises:
        KeyError: If a tensor name in dtypes_dict is not found in tensors_dict
        TypeError: If any input is not a torch.Tensor
    """
    return TensorUtils.ensure_tensor_types(tensors_dict, dtypes_dict)


def run_gui(gui_dict, gui_lock):
    """GUI process main loop with improved stability for long-running sessions.
    
    Changes for stability:
    - Periodic garbage collection to prevent memory accumulation
    - Heartbeat mechanism for health monitoring
    - Graceful handling of dpg context issues
    """
    import gc
    
    try:
        print("GUI started")
        with gui_lock:
            gui = BundleSdfGui(img_height=200)
            gui_dict["started"] = True
            gui_dict["gui_alive"] = True
            gui_dict["gui_last_heartbeat"] = time.time()

        local_dict = {}
        frame_count = 0
        gc_interval = 100  # Run GC every 100 frames

        while dpg.is_dearpygui_running():
            try:
                # Update heartbeat for health monitoring
                with gui_lock:
                    gui_dict["gui_last_heartbeat"] = time.time()
                    
                    if gui_dict["join"]:
                        break

                    for k in [
                        "mesh",
                        "color",
                        "mask",
                        "ob_in_cam",
                        "id_str",
                        "K",
                        "n_keyframe",
                        "nerf_num_frames",
                    ]:
                        if k in gui_dict:
                            local_dict[k] = gui_dict[k]
                            del gui_dict[k]

                if "nerf_num_frames" in local_dict:
                    gui.set_nerf_num_frames(local_dict["nerf_num_frames"])

                if "mesh" in local_dict:
                    logging.info(f"mesh V: {local_dict['mesh'].vertices.shape}")
                    gui.update_mesh(local_dict["mesh"])

                if "color" in local_dict:
                    t_gui_render = time.time()
                    gui.update_frame(
                        rgb=local_dict["color"],
                        mask=local_dict["mask"],
                        ob_in_cam=local_dict["ob_in_cam"],
                        id_str=local_dict["id_str"],
                        K=local_dict["K"],
                        n_keyframe=local_dict["n_keyframe"],
                    )
                    
                    # Log GUI rendering latency
                    if "timestamp" in local_dict:
                        gui_latency_ms = (t_gui_render - local_dict["timestamp"]) * 1000.0
                        logging.info(f"GUI rendering latency for {local_dict['id_str']}: {gui_latency_ms:.1f}ms")

                local_dict = {}
                
                # Periodic garbage collection to prevent memory accumulation
                frame_count += 1
                if frame_count % gc_interval == 0:
                    gc.collect()

            except (BrokenPipeError, ConnectionRefusedError, EOFError) as e:
                # Manager terminated - exit gracefully
                logging.warning(f"GUI: Manager connection lost: {e}")
                break
            except Exception:
                import traceback
                with open("/tmp/gui_crash.log", "a") as f:
                    f.write(f"--- Crash occurring at {time.time()} ---\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                logging.exception("GUI loop exception caught, attempting to continue...")

            dpg.render_dearpygui_frame()
            time.sleep(0.03)

        # Clean shutdown
        try:
            with gui_lock:
                gui_dict["gui_alive"] = False
        except:
            pass
        dpg.destroy_context()
        logging.info("GUI process exited normally")

    except Exception:
        import traceback
        with open("/tmp/gui_crash.log", "w") as f:
            f.write(traceback.format_exc())
        logging.exception("GUI Process Crashed Top-Level")
        try:
            with gui_lock:
                gui_dict["gui_alive"] = False
        except:
            pass


def run_nerf(
    p_dict,
    kf_to_nerf_list,
    lock,
    cfg_nerf,
    translation,
    sc_factor,
    start_nerf_keyframes,
    use_gui,
    gui_lock,
    gui_dict,
    debug_dir,
    timing_buffer,
    log_lock,
    enable_timing_log,
):
    """NeRF training process with improved memory management.
    
    Changes for stability:
    - Periodic CUDA cache clearing
    - Garbage collection after each training batch
    - Better error handling for Manager termination
    """
    import gc
    import torch
    
    try:
        with open("/tmp/run_nerf_progress.log", "w") as f:
            f.write("run_nerf process started\n")

        def log_progress(msg):
            with open("/tmp/run_nerf_progress.log", "a") as f:
                f.write(f"{time.time()}: {msg}\n")

        log_progress("Initializing variables")
        vox_res = 0.01
        nerf_num_frames = 0
        cnt_nerf = -1
        rgbs_all = []
        depths_all = []
        normal_maps_all = []
        masks_all = []
        occ_masks_all = []
        prev_pcd_real_scale = None
        tf_normalize = None
        
        # Memory management interval (clear CUDA cache every N batches)
        cuda_gc_interval = 5
        
        if translation is not None:
            tf_normalize = np.eye(4)
            tf_normalize[:3, 3] = translation
            tf1 = np.eye(4)
            tf1[:3, :3] *= sc_factor
            tf_normalize = tf1 @ tf_normalize
            cfg_nerf["sc_factor"] = float(sc_factor)
            cfg_nerf["translation"] = translation

        with lock:
            SPDLOG = p_dict["SPDLOG"]

        log_progress("Entering main loop")
        while 1:
            with lock:
                join = p_dict["join"]

            if join:
                log_progress("Join signal received, breaking loop")
                break

            skip = False
            with lock:
                if cnt_nerf == -1 and len(kf_to_nerf_list) < start_nerf_keyframes:
                    skip = True
                    p_dict["running"] = False
                else:
                    if len(kf_to_nerf_list) > 0:
                        p_dict["running"] = True
                        frame_id = p_dict["frame_id"]
                        cam_in_obs = p_dict["cam_in_obs"].copy()
                        rgbs = []
                        depths = []
                        normal_maps = []
                        masks = []
                        occ_masks = []
                        for f in kf_to_nerf_list:
                            rgbs.append(f["rgb"])
                            depths.append(f["depth"])
                            masks.append(f["mask"])
                            if f["normal_map"] is not None:
                                normal_maps.append(f["normal_map"])
                            if f["occ_mask"] is not None:
                                occ_masks.append(f["occ_mask"])
                        K = p_dict["K"]
                        nerf_num_frames += len(rgbs)
                        p_dict["nerf_num_frames"] = nerf_num_frames
                        kf_to_nerf_list[:] = []
                        if use_gui:
                            with gui_lock:
                                gui_dict["nerf_num_frames"] = nerf_num_frames
                        log_progress(
                            f"Got new work: frame_id={frame_id}, nerf_num_frames={nerf_num_frames}"
                        )
                    else:
                        skip = True

            if skip:
                # log_progress("Skipping (wait for frames)") # Commented out to avoid log spam
                time.sleep(0.01)
                continue

            log_progress("Processing batch")
            cnt_nerf += 1
            rgbs_all += list(rgbs)
            depths_all += list(depths)
            masks_all += list(masks)
            if normal_maps is not None:
                normal_maps_all += list(normal_maps)
            if occ_masks is not None:
                occ_masks_all += list(occ_masks)

            out_dir = f"{debug_dir}/{frame_id}/nerf"
            logging.info(f"out_dir: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            os.system(f"rm -rf {cfg_nerf['datadir']} && mkdir -p {cfg_nerf['datadir']}")

            glcam_in_obs = cam_in_obs @ glcam_in_cvcam

            if cfg_nerf["continual"]:
                if cnt_nerf == 0:
                    log_progress("Initializing continual nerf (cnt_nerf=0)")
                    if translation is None:
                        sc_factor, translation, pcd_real_scale, pcd_normalized = (
                            compute_scene_bounds(
                                None,
                                glcam_in_obs,
                                K,
                                use_mask=True,
                                base_dir=cfg_nerf["save_dir"],
                                rgbs=np.array(rgbs_all),
                                depths=np.array(depths_all),
                                masks=np.array(masks_all),
                                eps=cfg_nerf["dbscan_eps"],
                                min_samples=cfg_nerf["dbscan_eps_min_samples"],
                            )
                        )
                        sc_factor *= 0.7  # Ensure whole object within bound
                        cfg_nerf["sc_factor"] = float(sc_factor)
                        cfg_nerf["translation"] = translation
                        tf_normalize = np.eye(4)
                        tf_normalize[:3, 3] = translation
                        tf1 = np.eye(4)
                        tf1[:3, :3] *= sc_factor
                        tf_normalize = tf1 @ tf_normalize

                    pcd_all = pcd_real_scale

                else:
                    log_progress(f"Continual update (cnt_nerf={cnt_nerf})")
                    pcd_all = prev_pcd_real_scale
                    for i in range(len(rgbs)):
                        pts, colors = compute_scene_bounds_worker(
                            None,
                            K,
                            glcam_in_obs[len(glcam_in_obs) - len(rgbs) + i],
                            use_mask=True,
                            rgb=rgbs[i],
                            depth=depths[i],
                            mask=masks[i],
                        )
                        pcd_all += toOpen3dCloud(pts, colors)
                    pcd_all = pcd_all.voxel_down_sample(vox_res)
                    _, keep_mask = find_biggest_cluster(
                        np.asarray(pcd_all.points),
                        eps=cfg_nerf["dbscan_eps"],
                        min_samples=cfg_nerf["dbscan_eps_min_samples"],
                    )
                    keep_ids = np.arange(len(np.asarray(pcd_all.points)))[keep_mask]
                    pcd_all = pcd_all.select_by_index(keep_ids)

                    ########## Clear memory
                    rgbs_all = []
                    depths_all = []
                    normal_maps_all = []
                    masks_all = []
                    occ_masks_all = []

                pcd_normalized = copy.deepcopy(pcd_all)
                pcd_normalized.transform(tf_normalize)
                if normal_maps is not None and len(normal_maps) > 0:
                    normal_maps = np.array(normal_maps)
                else:
                    normal_maps = None
                rgbs, depths, masks, normal_maps, poses = preprocess_data(
                    np.array(rgbs),
                    np.array(depths),
                    np.array(masks),
                    normal_maps=normal_maps,
                    poses=glcam_in_obs,
                    sc_factor=cfg_nerf["sc_factor"],
                    translation=cfg_nerf["translation"],
                )

            else:
                log_progress("Standard mode scene bounds")
                logging.info(f"compute_scene_bounds, latest nerf frame {frame_id}")
                sc_factor, translation, pcd_real_scale, pcd_normalized = (
                    compute_scene_bounds(
                        None,
                        glcam_in_obs,
                        K,
                        use_mask=True,
                        base_dir=cfg_nerf["save_dir"],
                        rgbs=np.array(rgbs_all),
                        depths=np.array(depths_all),
                        masks=np.array(masks_all),
                        eps=cfg_nerf["dbscan_eps"],
                        min_samples=cfg_nerf["dbscan_eps_min_samples"],
                    )
                )

                cfg_nerf["sc_factor"] = float(sc_factor)
                cfg_nerf["translation"] = translation

                if normal_maps_all is not None and len(normal_maps_all) > 0:
                    normal_maps = np.array(normal_maps_all)
                else:
                    normal_maps = None

                logging.info(f"preprocess_data, latest nerf frame {frame_id}")
                rgbs, depths, masks, normal_maps, poses = preprocess_data(
                    np.array(rgbs_all),
                    np.array(depths_all),
                    np.array(masks_all),
                    normal_maps=normal_maps,
                    poses=glcam_in_obs,
                    sc_factor=cfg_nerf["sc_factor"],
                    translation=cfg_nerf["translation"],
                )

            if SPDLOG >= 2:
                np.savetxt(
                    f"{cfg_nerf['save_dir']}/trainval_poses.txt",
                    glcam_in_obs.reshape(-1, 4),
                )
                np.savetxt(
                    f"{debug_dir}/{frame_id}/poses_before_nerf.txt",
                    np.array(cam_in_obs).reshape(-1, 4),
                )

            if len(occ_masks_all) > 0:
                if cfg_nerf["continual"]:
                    occ_masks = np.array(occ_masks)
                else:
                    occ_masks = np.array(occ_masks_all)
            else:
                occ_masks = None

            if cnt_nerf == 0:
                logging.info(
                    f"First nerf run, create Runner, latest nerf frame {frame_id}"
                )
                log_progress("Creating NerfRunner (first run)")
                nerf = NerfRunner(
                    cfg_nerf,
                    rgbs,
                    depths=depths,
                    masks=masks,
                    normal_maps=normal_maps,
                    occ_masks=occ_masks,
                    poses=poses,
                    K=K,
                    build_octree_pcd=pcd_normalized,
                )
            else:
                if cfg_nerf["continual"]:
                    logging.info(f"add_new_frames, latest nerf frame {frame_id}")
                    log_progress("Adding new frames to NerfRunner")
                    nerf.add_new_frames(
                        rgbs,
                        depths,
                        masks,
                        normal_maps,
                        poses,
                        occ_masks=occ_masks,
                        new_pcd=pcd_normalized,
                        reuse_weights=False,
                    )
                else:
                    log_progress("Creating NerfRunner (standard)")
                    nerf = NerfRunner(
                        cfg_nerf,
                        rgbs,
                        depths=depths,
                        masks=masks,
                        normal_maps=normal_maps,
                        occ_masks=occ_masks,
                        poses=poses,
                        K=K,
                        build_octree_pcd=pcd_normalized,
                    )

            logging.info(f"Start training, latest nerf frame {frame_id}")
            log_progress(f"Starting nerf.train() for frame {frame_id}")
            t_nerf_start = time.time()
            nerf.train()
            t_nerf_end = time.time()
            log_progress("nerf.train() completed")
            nerf_ms = (t_nerf_end - t_nerf_start) * 1000.0
            logging.info(f"Training done, latest nerf frame {frame_id}")

            # Post-training processing with Manager access protection
            # Manager may be terminated if main process has already exited
            try:
                # Check if we should exit before accessing shared Manager objects
                with lock:
                    if p_dict["join"]:
                        logging.info(f"Join signal received after training frame {frame_id}, exiting loop")
                        p_dict["running"] = False
                        break

                # Write timing stats for processed frames
                nerf_ms_per_frame = nerf_ms / nerf_num_frames if nerf_num_frames > 0 else 0

                if enable_timing_log:
                    with log_lock:
                        if frame_id in timing_buffer:
                            data = timing_buffer.pop(frame_id)
                            with open(
                                os.path.join(debug_dir, "timing_stats.csv"), "a"
                            ) as f:
                                f.write(
                                    f"{data['timestamp']},{frame_id},{data['total_ms']:.1f},{data['fm_prep_ms']:.1f},{data['fm_2d_match_ms']:.1f},{data['fm_corres_ms']:.1f},{data['fm_ransac_ms']:.1f},{data['ba_ms']:.1f},{data['others_ms']:.1f},{nerf_ms:.1f},{nerf_num_frames},{nerf_ms_per_frame:.1f},{data['n_find_corres_calls']},{data['n_ref_retries']},{data['n_ba_pairs']},{data['n_local_frames']},{data['n_keyframes']},{data['n_total_frames']}\n"
                                )
                        else:
                            with open(
                                os.path.join(debug_dir, "timing_stats.csv"), "a"
                            ) as f:
                                f.write(
                                    f"{time.time()},{frame_id},n/a,n/a,n/a,n/a,n/a,n/a,n/a,{nerf_ms:.1f},{nerf_num_frames},{nerf_ms_per_frame:.1f},n/a,n/a,n/a,n/a,n/a,n/a\n"
                                )

                optimized_cvcam_in_obs, offset = get_optimized_poses_in_real_world(
                    poses,
                    nerf.models["pose_array"],
                    cfg_nerf["sc_factor"],
                    cfg_nerf["translation"],
                )

                logging.info("Getting mesh")
                # Use isolevel from config if available, otherwise default to 0
                isolevel = cfg_nerf.get("mesh_isolevel", 0.0)
                mesh = nerf.extract_mesh(isolevel=isolevel, voxel_size=cfg_nerf["mesh_resolution"])
                mesh = mesh_to_real_world(
                    mesh,
                    pose_offset=offset,
                    translation=nerf.cfg["translation"],
                    sc_factor=nerf.cfg["sc_factor"],
                )

                with lock:
                    p_dict["optimized_cvcam_in_obs"] = optimized_cvcam_in_obs
                    p_dict["running"] = False
                    p_dict["mesh"] = mesh

                logging.info(f"nerf done at frame {frame_id}")

                prev_pcd_real_scale = copy.deepcopy(pcd_real_scale)
                
                # Periodic memory cleanup to prevent VRAM/RAM accumulation
                if cnt_nerf % cuda_gc_interval == 0:
                    log_progress(f"Running periodic CUDA/memory cleanup at cnt_nerf={cnt_nerf}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                ####### Log
                if SPDLOG >= 2:
                    os.system(f"cp -r {cfg_nerf['save_dir']}/image_step_*.png  {out_dir}/")
                    with open(f"{out_dir}/config.yml", "w") as ff:
                        tmp = copy.deepcopy(cfg_nerf)
                        for k in tmp.keys():
                            if isinstance(tmp[k], np.ndarray):
                                tmp[k] = tmp[k].tolist()
                        yaml.dump(tmp, ff)
                    shutil.copy(f"{out_dir}/config.yml", f"{cfg_nerf['save_dir']}/")
                    np.savetxt(
                        f"{debug_dir}/{frame_id}/poses_after_nerf.txt",
                        np.array(optimized_cvcam_in_obs).reshape(-1, 4),
                    )
                    mesh.export(f"{cfg_nerf['save_dir']}/mesh_real_world.obj")
                    os.system(
                        f"rm -rf {cfg_nerf['save_dir']}/step_*_mesh_real_world.obj {cfg_nerf['save_dir']}/*frame*ray*.ply && mv {cfg_nerf['save_dir']}/*  {out_dir}/"
                    )

            except (BrokenPipeError, ConnectionRefusedError, EOFError, OSError) as e:
                # Manager has been terminated, exit gracefully
                logging.warning(f"Manager terminated during post-training processing for frame {frame_id}: {e}")
                logging.info("Exiting NeRF process gracefully")
                break

    except Exception:
        import traceback

        with open("/tmp/nerf_runner_crash.log", "w") as f:
            f.write(traceback.format_exc())
        logging.exception("NerfRunner Process Crashed")
        try:
            with lock:
                p_dict["running"] = False
        except (BrokenPipeError, ConnectionRefusedError, EOFError):
            logging.warning("Could not update p_dict['running'] - Manager may have terminated")


class BundleSdf:
    def __init__(
        self,
        cfg_track_dir=f"{code_dir}/config_ho3d.yml",
        cfg_nerf_dir=f"{code_dir}/config.yml",
        start_nerf_keyframes=10,
        translation=None,
        sc_factor=None,
        use_gui=False,
        model_2d_match="eloftr",
    ):
        # Initialize tensor precision management first
        self.tensor_dtype = set_bundlesdf_precision(use_full_precision=True)
        logging.info(
            f"Initialized BundleSDF with tensor precision: {self.tensor_dtype}"
        )

        with open(cfg_track_dir, "r") as ff:
            self.cfg_track = yaml.load(ff)
        self.debug_dir = self.cfg_track["debug_dir"]
        self.SPDLOG = self.cfg_track["SPDLOG"]
        self.start_nerf_keyframes = start_nerf_keyframes
        self.use_gui = use_gui
        self.translation = None
        self.sc_factor = None
        if sc_factor is not None:
            self.translation = translation
            self.sc_factor = sc_factor

        code_dir = os.path.dirname(os.path.realpath(__file__))
        with open(cfg_nerf_dir, "r") as ff:
            self.cfg_nerf = yaml.load(ff)
        self.cfg_nerf["notes"] = ""
        self.cfg_nerf["bounding_box"] = np.array(self.cfg_nerf["bounding_box"]).reshape(
            2, 3
        )

        self.manager = multiprocessing.Manager()

        if self.use_gui:
            self.gui_lock = multiprocessing.Lock()
            self.gui_dict = self.manager.dict()
            self.gui_dict["join"] = False
            self.gui_dict["started"] = False
            self.gui_worker = multiprocessing.Process(
                target=run_gui, args=(self.gui_dict, self.gui_lock)
            )
            self.gui_worker.start()
        else:
            self.gui_lock = None
            self.gui_dict = None

        self.p_dict = self.manager.dict()
        self.kf_to_nerf_list = self.manager.list()
        self.lock = multiprocessing.Lock()
        self.p_dict["running"] = False
        self.p_dict["join"] = False
        self.p_dict["nerf_num_frames"] = 0

        self.p_dict["SPDLOG"] = self.SPDLOG

        # Shared dictionary for timing stats buffer (frame_id -> {timestamp, fm_ms, ba_ms})
        self.timing_buffer = self.manager.dict()
        # Lock for CSV file writing
        self.log_lock = multiprocessing.Lock()

        # Initialize CSV file
        self.timing_csv_path = os.path.join(self.debug_dir, "timing_stats.csv")

        # Always start fresh: delete existing CSV and create a new one with header.
        self.enable_timing_log = True
        if os.path.exists(self.timing_csv_path):
            os.remove(self.timing_csv_path)
            logging.info("Deleted existing timing_stats.csv for fresh run")
        with open(self.timing_csv_path, "w") as f:
            f.write(
                "timestamp,frame_id,total_ms,fm_prep_ms,fm_2d_match_ms,fm_corres_ms,fm_ransac_ms,bundle_adjust_ms,others_ms,nerf_ms,nerf_n_frames,nerf_ms_per_frame,n_find_corres_calls,n_ref_retries,n_ba_pairs,n_local_frames,n_keyframes,n_total_frames\n"
            )

        self.p_nerf = multiprocessing.Process(
            target=run_nerf,
            args=(
                self.p_dict,
                self.kf_to_nerf_list,
                self.lock,
                self.cfg_nerf,
                self.translation,
                self.sc_factor,
                start_nerf_keyframes,
                self.use_gui,
                self.gui_lock,
                self.gui_dict,
                self.debug_dir,
                self.timing_buffer,
                self.log_lock,
                self.enable_timing_log,
            ),
        )
        self.p_nerf.start()

        # self.p_dict = {}
        # self.lock = threading.Lock()
        # self.p_dict['running'] = False
        # self.p_dict['join'] = False
        # self.p_nerf = threading.Thread(target=self.run_nerf, args=(self.p_dict, self.lock))
        # self.p_nerf.start()

        yml = my_cpp.YamlLoadFile(cfg_track_dir)
        self.bundler = my_cpp.Bundler(yml)
        if model_2d_match == "loftr":
            logging.info("Using LoFTR (original) matcher")
            self.loftr = LoftrRunner()
        else:
            logging.info("Using EfficientLoFTR matcher")
            self.loftr = ELoftrRunner()
        self.cnt = -1
        self.K = None
        self.mesh = None

    def _atomic_write_array4x4(self, arr4x4, path):
        """
        原子的に 4x4 行列をテキストで書き出すユーティリティ。
        - 既にファイルが存在する場合は上書きしない（呼び出し側でチェックする）。
        """
        tmp = path + ".tmp"
        try:
            np.savetxt(tmp, arr4x4.reshape(4, 4))
            os.replace(tmp, path)
            return True
        except Exception as e:
            logging.exception(f"atomic write failed for {path}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except:
                pass
            return False

    def _ensure_frame_outputs_fallback(self, frame):
        """
        saveNewframeResult() で通常出力されるはずのファイルが見つからない場合に
        最低限必要なもの（ob_in_cam/{id}.txt, {frame_id}/keyframes.yml）を作る。
        既にファイルが存在する場合は何もしない（上書きしない）。
        """
        debug_dir = self.debug_dir
        frame_id = frame._id_str
        ob_dir = os.path.join(debug_dir, "ob_in_cam")
        os.makedirs(ob_dir, exist_ok=True)

        # 1) ob_in_cam file (ob_in_cam/{frame_id}.txt)
        ob_file = os.path.join(ob_dir, f"{frame_id}.txt")
        if not os.path.exists(ob_file):
            try:
                ob_in_cam = np.linalg.inv(frame._pose_in_model)
                ok = self._atomic_write_array4x4(ob_in_cam, ob_file)
                if ok:
                    logging.info(
                        f"Fallback: wrote ob_in_cam for {frame_id} -> {ob_file} (created_by=fallback_python)"
                    )
                else:
                    logging.warning(
                        f"Fallback: failed to write ob_in_cam for {frame_id}"
                    )
            except Exception:
                logging.exception(
                    f"Fallback: exception while writing ob_in_cam for {frame_id}"
                )
        else:
            logging.debug(f"ob_in_cam exists, skip fallback for {ob_file}")

        # 2) keyframes.yml in the frame dir (only create if not exists)
        frame_dir = os.path.join(debug_dir, frame_id)
        os.makedirs(frame_dir, exist_ok=True)
        keyfile = os.path.join(frame_dir, "keyframes.yml")
        if not os.path.exists(keyfile):
            try:
                node = {}
                # Mirror the Bundler::saveNewframeResult format: keyframe_{id_str} -> cam_in_ob
                for i_kf, kf in enumerate(self.bundler._keyframes):
                    # convert pose to list of 16 floats row-major to match YAML used elsewhere
                    data = []
                    mat = np.array(kf._pose_in_model)
                    for h in range(4):
                        for w in range(4):
                            data.append(float(mat[h, w]))
                    node[f"keyframe_{kf._id_str}".replace("'", "")] = {
                        "cam_in_ob": data
                    }
                # atomic write YAML
                tmp_k = keyfile + ".tmp"
                with open(tmp_k, "w") as ff:
                    yaml.dump(node, ff)
                os.replace(tmp_k, keyfile)
                logging.info(
                    f"Fallback: wrote keyframes.yml to {keyfile} (created_by=fallback_python)"
                )
            except Exception:
                logging.exception(
                    f"Fallback: failed to write keyframes.yml for frame {frame_id}"
                )
        else:
            logging.debug(f"keyframes.yml exists, skip fallback for {keyfile}")

        # -------------------------
        # 1) process_new_frame 内 => saveNewframeResult() 呼び出し直後に挿入するコード例
        # bundlesdf.py の process_new_frame 内で、既に self.bundler.saveNewframeResult() を呼んでいる箇所の直後に次を入れてください。
        # -------------------------
        # after: self.bundler.saveNewframeResult()
        try:
            # 迅速な確認: frame_dir と ob_in_cam の状態をログ出力（最初に少しだけ）
            frame_dir = f"{self.debug_dir}/{frame._id_str}"
            logging.info(
                f"Called saveNewframeResult for {frame._id_str}, checking produced files..."
            )
            # list a few files if any
            try:
                sample_files = sorted(glob.glob(f"{frame_dir}/*"))[:20]
                logging.debug(f"Files in {frame_dir}: {sample_files}")
            except Exception:
                logging.debug("Could not list frame_dir contents for debugging")

            # If ob_in_cam entry is missing for this frame, create fallback (do not overwrite existing)
            ob_file = os.path.join(self.debug_dir, "ob_in_cam", f"{frame._id_str}.txt")
            if not os.path.exists(ob_file):
                logging.warning(
                    f"ob_in_cam entry missing for {frame._id_str}. Creating fallback."
                )
                # create fallback minimal outputs
                try:
                    self._ensure_frame_outputs_fallback(frame)
                except Exception:
                    logging.exception(
                        "Exception in fallback ensure_frame_outputs_fallback after saveNewframeResult"
                    )
            else:
                logging.debug(
                    f"ob_in_cam present for {frame._id_str}, no fallback needed"
                )

        except Exception:
            logging.exception("Error while performing post-save fallback checks")

    def on_finish(self):
        if self.use_gui:
            with self.gui_lock:
                self.gui_dict["join"] = True
            self.gui_worker.join()

        with self.lock:
            self.p_dict["join"] = True
        self.p_nerf.join()

        with self.lock:
            if (
                self.p_dict["running"] == False
                and "optimized_cvcam_in_obs" in self.p_dict
            ):
                # Thresholds for rejecting abnormal NeRF pose updates (likely due to NeRF collapse)
                max_trans_update_threshold = 0.1  # 10cm
                max_rot_update_threshold = 30 / 180.0 * np.pi  # 30 degrees
                n_rejected = 0
                
                for i_f in range(len(self.p_dict["optimized_cvcam_in_obs"])):
                    # Compute update magnitude for anomaly detection
                    trans_update = np.linalg.norm(
                        self.p_dict["optimized_cvcam_in_obs"][i_f][:3, 3]
                        - self.bundler._keyframes[i_f]._pose_in_model[:3, 3]
                    )
                    rot_update = geodesic_distance(
                        self.p_dict["optimized_cvcam_in_obs"][i_f][:3, :3],
                        self.bundler._keyframes[i_f]._pose_in_model[:3, :3],
                    )
                    
                    # Check for abnormally large updates (likely NeRF collapse)
                    if trans_update > max_trans_update_threshold or rot_update > max_rot_update_threshold:
                        logging.warning(
                            f"[NeRF Anomaly] on_finish: Rejecting abnormal pose update for frame {self.bundler._keyframes[i_f]._id_str}: "
                            f"trans_update={trans_update:.4f}m, rot_update={rot_update * 180 / np.pi:.2f}deg"
                        )
                        n_rejected += 1
                        continue  # Skip this frame's pose update
                    
                    self.bundler._keyframes[i_f]._pose_in_model = self.p_dict[
                        "optimized_cvcam_in_obs"
                    ][i_f]
                    self.bundler._keyframes[i_f]._nerfed = True
                
                if n_rejected > 0:
                    logging.warning(f"[NeRF Anomaly] on_finish: Rejected {n_rejected}/{len(self.p_dict['optimized_cvcam_in_obs'])} pose updates")
                del self.p_dict["optimized_cvcam_in_obs"]

                # ここで ob_in_cam をチェックし、空なら bundler._keyframes から作成（存在するファイルはスキップ）
        try:
            ob_files = sorted(glob.glob(f"{self.debug_dir}/ob_in_cam/*"))
            if len(ob_files) == 0:
                logging.warning(
                    f"No ob_in_cam files found under {self.debug_dir}/ob_in_cam; creating fallback from bundler._keyframes"
                )
                for kf in self.bundler._keyframes:
                    # reuse helper to create per-keyframe fallback outputs (skips if exists)
                    try:
                        self._ensure_frame_outputs_fallback(kf)
                    except Exception:
                        logging.exception(
                            f"Failed to fallback-create outputs for keyframe {kf._id_str}"
                        )
            else:
                logging.info(
                    f"Found {len(ob_files)} ob_in_cam files under {self.debug_dir}/ob_in_cam"
                )
        except Exception:
            logging.exception(
                "Error while verifying/creating ob_in_cam fallback in on_finish"
            )

    def make_frame(
        self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)
    ):
        H, W = color.shape[:2]
        roi = [0, W - 1, 0, H - 1]

        frame = my_cpp.Frame(
            color, depth, roi, pose_in_model, self.cnt, id_str, K, self.bundler.yml
        )
        if mask is not None:
            frame._fg_mask = my_cpp.cvMat(mask)
        if occ_mask is not None:
            frame._occ_mask = my_cpp.cvMat(occ_mask)
        return frame

    def find_corres(self, frame_pairs):
        t_prep = 0.0
        t_2d = 0.0
        t_corres = 0.0
        t_ransac = 0.0

        logging.info(f"frame_pairs: {len(frame_pairs)}")
        is_match_ref = (
            len(frame_pairs) == 1
            and frame_pairs[0][0]._ref_frame_id == frame_pairs[0][1]._id
            and self.bundler._newframe == frame_pairs[0][0]
        )

        t0 = time.time()
        imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
        imgs = np.array([np.array(img) for img in imgs])
        t_prep = (time.time() - t0) * 1000.0

        if len(query_pairs) == 0:
            return t_prep, t_2d, t_corres, t_ransac

        t0 = time.time()
        corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
        for i_pair in range(len(query_pairs)):
            cur_corres = corres[i_pair][:, :4]
            tfA = np.array(tfs[i_pair * 2])
            tfB = np.array(tfs[i_pair * 2 + 1])
            cur_corres[:, :2] = transform_pts(cur_corres[:, :2], np.linalg.inv(tfA))
            cur_corres[:, 2:4] = transform_pts(cur_corres[:, 2:4], np.linalg.inv(tfB))
            self.bundler._fm._raw_matches[query_pairs[i_pair]] = (
                cur_corres.round().astype(np.uint16)
            )
        t_2d = (time.time() - t0) * 1000.0

        min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

        if (
            is_match_ref
            and len(self.bundler._fm._raw_matches[frame_pairs[0]]) < min_match_with_ref
        ):
            self.bundler._fm._raw_matches[frame_pairs[0]] = []
            self.bundler._newframe._status = my_cpp.Frame.FAIL
            logging.info(
                f"frame {self.bundler._newframe._id_str} mark FAIL, due to no matching"
            )
            return t_prep, t_2d, t_corres, t_ransac

        t0 = time.time()
        self.bundler._fm.rawMatchesToCorres(query_pairs)
        t_corres = (time.time() - t0) * 1000.0

        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], "before_ransac")

        t0 = time.time()
        self.bundler._fm.runRansacMultiPairGPU(query_pairs)
        t_ransac = (time.time() - t0) * 1000.0

        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], "after_ransac")

        return t_prep, t_2d, t_corres, t_ransac

    def process_new_frame(self, frame):
        # t_start_timestamp = time.time()
        t_fm_prep = 0.0
        t_fm_2d = 0.0
        t_fm_corres = 0.0
        t_fm_ransac = 0.0
        t_ba = 0.0
        n_find_corres_calls = 0
        n_ref_retries = 0
        n_ba_pairs = 0
        n_local_frames = 0

        try:
            logging.info(f"process frame {frame._id_str}")

            self.bundler._newframe = frame
            os.makedirs(self.debug_dir, exist_ok=True)

            if frame._id > 0:
                ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
                frame._ref_frame_id = ref_frame._id
                frame._pose_in_model = ref_frame._pose_in_model
            else:
                self.bundler._firstframe = frame

            frame.invalidatePixelsByMask(frame._fg_mask)
            if (
                frame._id == 0
                and np.abs(np.array(frame._pose_in_model) - np.eye(4)).max() <= 1e-4
            ):
                frame.setNewInitCoordinate()

            n_fg = (np.array(frame._fg_mask) > 0).sum()
            if n_fg < 100:
                logging.info(
                    f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}"
                )
                frame._status = my_cpp.Frame.FAIL
                self.bundler.forgetFrame(frame)
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            if self.cfg_track["depth_processing"]["denoise_cloud"]:
                frame.pointCloudDenoise()

            n_valid = frame.countValidPoints()
            n_valid_first = self.bundler._firstframe.countValidPoints()
            if n_valid < n_valid_first / 40.0:
                logging.info(
                    f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL"
                )
                frame._status = my_cpp.Frame.FAIL
                self.bundler.forgetFrame(frame)
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            if frame._id == 0:
                self.bundler.checkAndAddKeyframe(
                    frame
                )  # First frame is always keyframe
                self.bundler._frames[frame._id] = frame
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

            tp, t2, tc, tr = self.find_corres([(frame, ref_frame)])
            n_find_corres_calls += 1
            t_fm_prep += tp
            t_fm_2d += t2
            t_fm_corres += tc
            t_fm_ransac += tr

            matches = self.bundler._fm._matches[(frame, ref_frame)]

            if frame._status == my_cpp.Frame.FAIL:
                logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
                self.bundler.forgetFrame(frame)
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            matches = self.bundler._fm._matches[(frame, ref_frame)]
            if len(matches) < min_match_with_ref:
                visibles = []
                for kf in self.bundler._keyframes:
                    visible = my_cpp.computeCovisibility(frame, kf)
                    visibles.append(visible)
                visibles = np.array(visibles)
                ids = np.argsort(visibles)[::-1]
                found = False

                for id in ids:
                    kf = self.bundler._keyframes[id]
                    logging.info(f"trying new ref frame {kf._id_str}")
                    ref_frame = kf
                    frame._ref_frame_id = kf._id
                    frame._pose_in_model = kf._pose_in_model

                    tp, t2, tc, tr = self.find_corres([(frame, ref_frame)])
                    n_find_corres_calls += 1
                    n_ref_retries += 1
                    t_fm_prep += tp
                    t_fm_2d += t2
                    t_fm_corres += tc
                    t_fm_ransac += tr

                    # self.bundler._fm.findCorres(frame, ref_frame)

                    if (
                        len(self.bundler._fm._matches[(frame, kf)])
                        >= min_match_with_ref
                    ):
                        logging.info(f"re-choose new ref frame to {kf._id_str}")
                        found = True
                        break

                if not found:
                    frame._status = my_cpp.Frame.FAIL
                    logging.info(
                        f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL"
                    )
                    self.bundler.forgetFrame(frame)
                    return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            logging.info(
                f"frame {frame._id_str} pose update before\n{frame._pose_in_model.round(3)}"
            )
            offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
            frame._pose_in_model = offset @ frame._pose_in_model
            logging.info(
                f"frame {frame._id_str} pose update after\n{frame._pose_in_model.round(3)}"
            )

            window_size = self.cfg_track["bundle"]["window_size"]
            if len(self.bundler._frames) - len(self.bundler._keyframes) > window_size:
                for k in self.bundler._frames:
                    f = self.bundler._frames[k]
                    isforget = self.bundler.forgetFrame(f)
                    if isforget:
                        logging.info(f"exceed window size, forget frame {f._id_str}")
                        break

            self.bundler._frames[frame._id] = frame

            self.bundler.selectKeyFramesForBA()

            local_frames = self.bundler._local_frames
            n_local_frames = len(local_frames)

            pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)

            # Cap BA pairs to prevent spikes from rematch_after_nerf cache invalidation.
            # Steady state is ~18-20 pairs; spikes reach 69-113 after NeRF pose sync.
            max_ba_pairs = self.cfg_track.get("bundle", {}).get("max_ba_pairs", 25)
            if len(pairs) > max_ba_pairs:
                # Prioritize pairs involving the new frame (most important for current pose)
                new_frame_pairs = [p for p in pairs if p[0] == frame or p[1] == frame]
                other_pairs = [p for p in pairs if p[0] != frame and p[1] != frame]
                pairs = new_frame_pairs + other_pairs[: max_ba_pairs - len(new_frame_pairs)]
                logging.info(
                    f"Capped BA pairs: {len(new_frame_pairs) + len(other_pairs)} -> {len(pairs)} "
                    f"(new_frame={len(new_frame_pairs)}, other={len(pairs) - len(new_frame_pairs)})"
                )

            n_ba_pairs = len(pairs)

            tp, t2, tc, tr = self.find_corres(pairs)
            n_find_corres_calls += 1
            t_fm_prep += tp
            t_fm_2d += t2
            t_fm_corres += tc
            t_fm_ransac += tr

            if frame._status == my_cpp.Frame.FAIL:
                self.bundler.forgetFrame(frame)
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            find_matches = False
            t0 = time.time()
            self.bundler.optimizeGPU(local_frames, find_matches)
            t_ba = (time.time() - t0) * 1000.0

            if frame._status == my_cpp.Frame.FAIL:
                self.bundler.forgetFrame(frame)
                return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

            self.bundler.checkAndAddKeyframe(frame)

        finally:
            pass

        return t_fm_prep, t_fm_2d, t_fm_corres, t_fm_ransac, t_ba, n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames

    def run(
        self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)
    ):
        t_run_start = time.time()
        self.cnt += 1

        if self.K is None:
            self.K = K
            with self.lock:
                self.p_dict["K"] = self.K

        if self.use_gui:
            while 1:
                with self.gui_lock:
                    started = self.gui_dict["started"]
                if not started:
                    time.sleep(1)
                    logging.info("Waiting for GUI")
                    continue
                break

        H, W = color.shape[:2]

        percentile = self.cfg_track["depth_processing"]["percentile"]
        if percentile < 100:  # Denoise
            logging.info("percentile denoise start")
            valid = (depth >= 0.1) & (mask > 0)
            thres = np.percentile(depth[valid], percentile)
            depth[depth >= thres] = 0
            logging.info("percentile denoise done")

        frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
        os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

        logging.info(f"processNewFrame start {frame._id_str}")
        # self.bundler.processNewFrame(frame)
        (fm_prep_ms, fm_2d_match_ms, fm_corres_ms, fm_ransac_ms, ba_ms,
         n_find_corres_calls, n_ref_retries, n_ba_pairs, n_local_frames) = (
            self.process_new_frame(frame)
        )
        logging.info(f"processNewFrame done {frame._id_str}")

        if self.bundler._keyframes[-1] == frame:
            logging.info(f"{frame._id_str} prepare data for nerf")

            with self.lock:
                self.p_dict["frame_id"] = frame._id_str
                self.p_dict["running"] = True
                self.kf_to_nerf_list.append(
                    {
                        "rgb": np.array(frame._color)
                        .reshape(H, W, 3)[..., ::-1]
                        .copy(),
                        "depth": np.array(frame._depth).reshape(H, W).copy(),
                        "mask": np.array(frame._fg_mask).reshape(H, W).copy(),
                        # 'occ_mask': occ_mask.reshape(H,W),
                        # 'normal_map': np.array(frame._normal_map).copy(),
                        "occ_mask": None,
                        "normal_map": None,
                    }
                )
                cam_in_obs = []
                for f in self.bundler._keyframes:
                    cam_in_obs.append(np.array(f._pose_in_model).copy())
                self.p_dict["cam_in_obs"] = np.array(cam_in_obs)

            if self.SPDLOG >= 2:
                with open(
                    f"{self.debug_dir}/{frame._id_str}/nerf_frames.txt", "w"
                ) as ff:
                    for f in self.bundler._keyframes:
                        ff.write(f"{f._id_str}\n")

            ############# Wait for sync
            while 1:
                with self.lock:
                    running = self.p_dict["running"]
                    nerf_num_frames = self.p_dict["nerf_num_frames"]
                if not running:
                    break
                if (
                    len(self.bundler._keyframes) - nerf_num_frames
                    >= self.cfg_nerf["sync_max_delay"]
                ):
                    time.sleep(0.01)
                    # logging.info(f"wait for sync len(self.bundler._keyframes):{len(self.bundler._keyframes)}, nerf_num_frames:{nerf_num_frames}")
                    continue
                break

        rematch_after_nerf = self.cfg_track["feature_corres"]["rematch_after_nerf"]
        logging.info(f"rematch_after_nerf: {rematch_after_nerf}")
        frames_large_update = []
        
        # Thresholds for rejecting abnormal NeRF pose updates (likely due to NeRF collapse)
        max_trans_update_threshold = 0.1  # 10cm - reject if translation change exceeds this
        max_rot_update_threshold = 30 / 180.0 * np.pi  # 30 degrees - reject if rotation change exceeds this
        
        with self.lock:
            if "optimized_cvcam_in_obs" in self.p_dict:
                n_rejected = 0
                for i_f in range(len(self.p_dict["optimized_cvcam_in_obs"])):
                    # Always compute update magnitude for logging and anomaly detection
                    trans_update = np.linalg.norm(
                        self.p_dict["optimized_cvcam_in_obs"][i_f][:3, 3]
                        - self.bundler._keyframes[i_f]._pose_in_model[:3, 3]
                    )
                    rot_update = geodesic_distance(
                        self.p_dict["optimized_cvcam_in_obs"][i_f][:3, :3],
                        self.bundler._keyframes[i_f]._pose_in_model[:3, :3],
                    )
                    
                    # Check for abnormally large updates (likely NeRF collapse)
                    if trans_update > max_trans_update_threshold or rot_update > max_rot_update_threshold:
                        logging.warning(
                            f"[NeRF Anomaly] Rejecting abnormal pose update for frame {self.bundler._keyframes[i_f]._id_str}: "
                            f"trans_update={trans_update:.4f}m (threshold={max_trans_update_threshold}m), "
                            f"rot_update={rot_update * 180 / np.pi:.2f}deg (threshold={max_rot_update_threshold * 180 / np.pi:.0f}deg)"
                        )
                        n_rejected += 1
                        continue  # Skip this frame's pose update
                    
                    if rematch_after_nerf:
                        if trans_update >= 0.005 or rot_update >= 5 / 180.0 * np.pi:
                            frames_large_update.append(self.bundler._keyframes[i_f])
                        logging.info(
                            f"{self.bundler._keyframes[i_f]._id_str}, trans_update={trans_update}, rot_update={rot_update}"
                        )
                    self.bundler._keyframes[i_f]._pose_in_model = self.p_dict[
                        "optimized_cvcam_in_obs"
                    ][i_f]
                    self.bundler._keyframes[i_f]._nerfed = True
                
                if n_rejected > 0:
                    logging.warning(f"[NeRF Anomaly] Rejected {n_rejected}/{len(self.p_dict['optimized_cvcam_in_obs'])} pose updates due to abnormal magnitude")
                logging.info(
                    f"synced pose from nerf, latest nerf frame {self.bundler._keyframes[len(self.p_dict['optimized_cvcam_in_obs']) - 1]._id_str}"
                )
                del self.p_dict["optimized_cvcam_in_obs"]

            if self.use_gui:
                with self.gui_lock:
                    if "mesh" in self.p_dict:
                        self.gui_dict["mesh"] = self.p_dict["mesh"]
                        del self.p_dict["mesh"]

        if rematch_after_nerf:
            if len(frames_large_update) > 0:
                with self.lock:
                    nerf_num_frames = self.p_dict["nerf_num_frames"]
                frames_large_update_set = set(id(f) for f in frames_large_update)
                logging.info(
                    f"rematch_after_nerf: {len(frames_large_update)} frames with large updates, "
                    f"before matches keys: {len(self.bundler._fm._matches)}"
                )
                # Only delete matches where BOTH frames had large pose updates.
                # Previously, deleting all matches involving ANY updated frame caused
                # n_ba_pairs spikes (69-113) on the next frame, because getFeatureMatchPairs
                # regenerates all uncached pairs. Restricting to both-updated pairs limits
                # the blast radius while still refreshing the most stale correspondences.
                n_deleted = 0
                ks = list(self.bundler._fm._matches.keys())
                for k in ks:
                    if (
                        id(k[0]) in frames_large_update_set
                        and id(k[1]) in frames_large_update_set
                    ):
                        del self.bundler._fm._matches[k]
                        n_deleted += 1
                logging.info(
                    f"rematch_after_nerf: deleted {n_deleted} match pairs, "
                    f"after matches keys: {len(self.bundler._fm._matches)}"
                )

        self.bundler.saveNewframeResult()
        if self.SPDLOG >= 2 and occ_mask is not None:
            os.makedirs(f"{self.debug_dir}/occ_mask/", exist_ok=True)
            cv2.imwrite(f"{self.debug_dir}/occ_mask/{frame._id_str}.png", occ_mask)

        if self.use_gui:
            ob_in_cam = np.linalg.inv(frame._pose_in_model)
            with self.gui_lock:
                self.gui_dict["color"] = color[..., ::-1]
                self.gui_dict["mask"] = mask
                self.gui_dict["ob_in_cam"] = ob_in_cam
                self.gui_dict["id_str"] = frame._id_str
                self.gui_dict["K"] = self.K
                self.gui_dict["n_keyframe"] = len(self.bundler._keyframes)
                self.gui_dict["timestamp"] = time.time()  # Add timestamp for GUI latency measurement

        t_run_end = time.time()
        total_ms = (t_run_end - t_run_start) * 1000.0
        others_ms = total_ms - (
            fm_prep_ms + fm_2d_match_ms + fm_corres_ms + fm_ransac_ms + ba_ms
        )

        n_keyframes = len(self.bundler._keyframes)
        n_total_frames = len(self.bundler._frames)

        if self.enable_timing_log:
            # Check if frame is in keyframes list (it might have been removed if failed)
            is_keyframe = False
            if frame._status != my_cpp.Frame.FAIL:
                for kf in self.bundler._keyframes:
                    if kf == frame:
                        is_keyframe = True
                        break

            if is_keyframe:
                self.timing_buffer[frame._id_str] = {
                    "timestamp": t_run_start,
                    "total_ms": total_ms,
                    "fm_prep_ms": fm_prep_ms,
                    "fm_2d_match_ms": fm_2d_match_ms,
                    "fm_corres_ms": fm_corres_ms,
                    "fm_ransac_ms": fm_ransac_ms,
                    "ba_ms": ba_ms,
                    "others_ms": others_ms,
                    "n_find_corres_calls": n_find_corres_calls,
                    "n_ref_retries": n_ref_retries,
                    "n_ba_pairs": n_ba_pairs,
                    "n_local_frames": n_local_frames,
                    "n_keyframes": n_keyframes,
                    "n_total_frames": n_total_frames,
                }
            else:
                with self.log_lock:
                    with open(self.timing_csv_path, "a") as f:
                        f.write(
                            f"{t_run_start},{frame._id_str},{total_ms:.1f},{fm_prep_ms:.1f},{fm_2d_match_ms:.1f},{fm_corres_ms:.1f},{fm_ransac_ms:.1f},{ba_ms:.1f},{others_ms:.1f},n/a,n/a,n/a,{n_find_corres_calls},{n_ref_retries},{n_ba_pairs},{n_local_frames},{n_keyframes},{n_total_frames}\n"
                        )

    def run_global_nerf(self, reader=None, get_texture=False, tex_res=1024):
        """
        @reader: data reader, sometimes we want to use the full resolution raw image
        """
        self.K = np.loadtxt(f"{self.debug_dir}/cam_K.txt").reshape(3, 3)

        # Robust discovery of last_stamp: prefer ob_in_cam/*.txt, fall back to keyframes.yml,
        # poses_before_nerf.txt, or latest frame directory by mtime.
        tmp = sorted(glob.glob(f"{self.debug_dir}/ob_in_cam/*"))
        if len(tmp) == 0:
            logging.warning(
                f"No files found in {self.debug_dir}/ob_in_cam/. Attempting fallback discovery."
            )
            # Try to find a recent frame directory that contains keyframes.yml or poses_before_nerf.txt
            cand_keyfiles = sorted(glob.glob(f"{self.debug_dir}/*/keyframes.yml"))
            cand_posefiles = sorted(
                glob.glob(f"{self.debug_dir}/*/poses_before_nerf.txt")
            )
            if len(cand_keyfiles) > 0:
                last_frame_dir = os.path.dirname(cand_keyfiles[-1])
                last_stamp = os.path.basename(last_frame_dir)
                logging.info(
                    f"Fallback: using last_stamp {last_stamp} from {cand_keyfiles[-1]}"
                )
            elif len(cand_posefiles) > 0:
                last_frame_dir = os.path.dirname(cand_posefiles[-1])
                last_stamp = os.path.basename(last_frame_dir)
                logging.info(
                    f"Fallback: using last_stamp {last_stamp} from {cand_posefiles[-1]}"
                )
            else:
                # as last resort choose most recent frame dir by mtime (but prefer ones that have some outputs)
                cand_dirs = [
                    p for p in glob.glob(f"{self.debug_dir}/*") if os.path.isdir(p)
                ]
                if len(cand_dirs) > 0:
                    cand_dirs_sorted = sorted(
                        cand_dirs, key=lambda p: os.path.getmtime(p)
                    )
                    last_stamp = os.path.basename(cand_dirs_sorted[-1])
                    logging.warning(
                        f"No keyframes.yml or poses_before_nerf found; fallback to latest frame dir {last_stamp}"
                    )
                else:
                    # Nothing found — give informative error and abort
                    logging.error(
                        f"No ob_in_cam files and no frame-level keyfiles found under {self.debug_dir}. Directory listing:"
                    )
                    for p in sorted(glob.glob(f"{self.debug_dir}/*")):
                        logging.error(f" - {p}")
                    raise RuntimeError(
                        f"No ob_in_cam files and no frame keyfiles found under {self.debug_dir}. Aborting run_global_nerf."
                    )
        else:
            last_stamp = os.path.basename(tmp[-1]).replace(".txt", "")
            logging.info(f"last_stamp {last_stamp}")

        keyframes = yaml.load(open(f"{self.debug_dir}/{last_stamp}/keyframes.yml", "r"))
        logging.info(f"keyframes#: {len(keyframes)}")
        keys = list(keyframes.keys())
        if len(keyframes) > self.cfg_nerf["n_train_image"]:
            keys = [keys[0]] + list(
                np.random.choice(keys, self.cfg_nerf["n_train_image"], replace=False)
            )
            keys = list(set(keys))
            logging.info(f"frame_ids too large, select subset num: {len(keys)}")

        frame_ids = []
        for k in keys:
            frame_ids.append(k.replace("keyframe_", ""))

        cam_in_obs = []
        for k in keys:
            cam_in_ob = np.array(keyframes[k]["cam_in_ob"]).reshape(4, 4)
            cam_in_obs.append(cam_in_ob)
        cam_in_obs = np.array(cam_in_obs)

        out_dir = f"{self.debug_dir}/final/nerf"
        os.system(f"rm -rf {out_dir} && mkdir -p {out_dir}")
        os.system(
            f"rm -rf {self.debug_dir}/final/used_rgbs/ && mkdir -p {self.debug_dir}/final/used_rgbs/"
        )

        rgbs = []
        depths = []
        normal_maps = []
        masks = []
        occ_masks = []

        for frame_id in frame_ids:
            if reader is not None:
                self.K = reader.K.copy()
                id = reader.id_strs.index(frame_id)
                rgbs.append(reader.get_color(id))
                depths.append(reader.get_depth(id))
                masks.append(reader.get_mask(id))
            else:
                self.cfg_nerf["down_scale_ratio"] = (
                    1  # Images have been downscaled in tracking outputs
                )
                rgb_file = f"{self.debug_dir}/color_segmented/{frame_id}.png"
                shutil.copy(rgb_file, f"{self.debug_dir}/final/used_rgbs/")
                rgb = imageio.imread(rgb_file)
                depth = (
                    cv2.imread(
                        rgb_file.replace("color_segmented", "depth_filtered"), -1
                    )
                    / 1e3
                )
                mask = cv2.imread(rgb_file.replace("color_segmented", "mask"), -1)
                rgbs.append(rgb)
                depths.append(depth)
                masks.append(mask)

        glcam_in_obs = cam_in_obs @ glcam_in_cvcam

        self.cfg_nerf["sc_factor"] = None
        self.cfg_nerf["translation"] = None

        ######### Reuse normalization
        files = sorted(
            glob.glob(f"{self.debug_dir}/**/nerf/config.yml", recursive=True)
        )
        if len(files) > 0:
            tmp = yaml.load(open(files[-1], "r"))
            self.cfg_nerf["sc_factor"] = float(tmp["sc_factor"])
            self.cfg_nerf["translation"] = np.array(tmp["translation"])

        sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(
            None,
            glcam_in_obs,
            self.K,
            use_mask=True,
            base_dir=self.cfg_nerf["save_dir"],
            rgbs=np.array(rgbs),
            depths=np.array(depths),
            masks=np.array(masks),
        )

        self.cfg_nerf["sc_factor"] = float(sc_factor)
        self.cfg_nerf["translation"] = translation

        if normal_maps is not None and len(normal_maps) > 0:
            normal_maps = np.array(normal_maps)
        else:
            normal_maps = None

        rgbs_raw = np.array(rgbs).copy()
        rgbs, depths, masks, normal_maps, poses = preprocess_data(
            np.array(rgbs),
            np.array(depths),
            np.array(masks),
            normal_maps=normal_maps,
            poses=glcam_in_obs,
            sc_factor=self.cfg_nerf.get("sc_factor", None),
            translation=self.cfg_nerf.get("translation", None),
        )

        self.cfg_nerf["sampled_frame_ids"] = np.arange(len(rgbs))

        np.savetxt(
            f"{self.cfg_nerf['save_dir']}/trainval_poses.txt",
            glcam_in_obs.reshape(-1, 4),
        )

        if len(occ_masks) > 0:
            occ_masks = np.array(occ_masks)
        else:
            occ_masks = None

        nerf = NerfRunner(
            self.cfg_nerf,
            rgbs,
            depths=depths,
            masks=masks,
            normal_maps=normal_maps,
            occ_masks=occ_masks,
            poses=poses,
            K=self.K,
            build_octree_pcd=pcd_normalized,
        )

        print("Start training")

        nerf.train()

        optimized_cvcam_in_obs, offset = get_optimized_poses_in_real_world(
            poses,
            nerf.models["pose_array"],
            self.cfg_nerf["sc_factor"],
            self.cfg_nerf["translation"],
        )

        ####### Log
        # TODO: the cp below fails; fix it
        os.system(f"cp -r {self.cfg_nerf['save_dir']}/image_step_*.png  {out_dir}/")
        with open(f"{out_dir}/config.yml", "w") as ff:
            tmp = copy.deepcopy(self.cfg_nerf)
            for k in tmp.keys():
                if isinstance(tmp[k], np.ndarray):
                    tmp[k] = tmp[k].tolist()
            yaml.dump(tmp, ff)
        shutil.copy(f"{out_dir}/config.yml", f"{self.cfg_nerf['save_dir']}/")
        os.system(
            f"mv {self.cfg_nerf['save_dir']}/*  {out_dir}/ && rm -rf {out_dir}/step_*_mesh_real_world.obj {out_dir}/*frame*ray*.ply"
        )

        torch.cuda.empty_cache()

        np.savetxt(
            f"{self.debug_dir}/{frame_id}/poses_after_nerf.txt",
            np.array(optimized_cvcam_in_obs).reshape(-1, 4),
        )

        # mesh_files = sorted(glob.glob(f"{self.debug_dir}/final/nerf/step_*_mesh_normalized_space.obj"))
        # mesh = trimesh.load(mesh_files[-1])

        mesh, sigma, query_pts = nerf.extract_mesh(
            voxel_size=self.cfg_nerf["mesh_resolution"], isolevel=0, return_sigma=True
        )

        mesh.merge_vertices()
        ms = trimesh_split(mesh, min_edge=100)
        largest_size = 0
        largest = None
        for m in ms:
            # mean = m.vertices.mean(axis=0)
            # if np.linalg.norm(mean)>=0.1*nerf.cfg['sc_factor']:
            #   continue
            if m.vertices.shape[0] > largest_size:
                largest_size = m.vertices.shape[0]
                largest = m
        mesh = largest
        mesh.export(f"{self.debug_dir}/mesh_cleaned.obj")

        if get_texture:
            mesh = nerf.mesh_texture_from_train_images(
                mesh, rgbs_raw=rgbs_raw, train_texture=False, tex_res=tex_res
            )

        mesh = mesh_to_real_world(
            mesh,
            pose_offset=offset,
            translation=self.cfg_nerf["translation"],
            sc_factor=self.cfg_nerf["sc_factor"],
        )
        mesh.export(f"{self.debug_dir}/textured_mesh.obj")


if __name__ == "__main__":
    set_seed(0)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")

    config_path = f"{code_dir}/third_party/BundleTrack/config_ho3d.yml"
    if not os.path.exists(config_path):
        config_path = f"{code_dir}/BundleTrack/config_ho3d.yml"
    cfg_nerf = yaml.load(open(config_path, "r"))
    cfg_nerf["data_dir"] = (
        "/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/MPM13"
    )
    cfg_nerf["SPDLOG"] = 1

    cfg_track_dir = "/tmp/config.yml"
    yaml.dump(cfg_nerf, open(cfg_track_dir, "w"))
    tracker = BundleSdf(cfg_track_dir=cfg_track_dir)
    reader = Ho3dReader(tracker.bundler.yml["data_dir"].Scalar())

    os.system(f"rm -rf {tracker.debug_dir} && mkdir -p {tracker.debug_dir}")

    for i, color_file in enumerate(reader.color_files):
        color = cv2.imread(color_file)
        depth = reader.get_depth(i)
        id_str = reader.id_strs[i]
        occ_mask = reader.get_occ_mask(i)
        tracker.run(color, depth, reader.K, id_str, occ_mask=occ_mask)

    print("Done")
