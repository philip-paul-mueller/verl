# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        if config.async_training.use_trainer_do_validate:
            # Serial, trainer-first: the rollouter needs the trainer's hybrid worker
            # group injected before its init_workers(), so the trainer must init first.
            print("[ASYNC MAIN] Creating FullyAsyncTrainer first (needed for hybrid worker group injection)...")
            self._create_trainer(config)

            # Extract the trainer's actor_rollout_wg to inject into the rollouter; it backs
            # the hybrid rollout replicas used during trainer-side validation.
            print("[ASYNC MAIN] Injecting trainer's worker group into rollouter for hybrid replicas...")
            trainer_wg = ray.get(self.components["trainer"].get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )

            print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
            self._create_rollouter(config)
        else:
            # No hybrid worker group coupling: trainer and rollouter are independent.
            # Overlap their init_workers() so the two disk loads run in parallel instead
            # of trainer-then-rollouter.
            print("[ASYNC MAIN] use_trainer_do_validate=False: initializing trainer and rollouter in parallel...")
            trainer = self._build_trainer(config)
            rollouter = self._build_rollouter(config)
            ray.get([trainer.init_workers.remote(), rollouter.init_workers.remote()])
            self.components["trainer"] = trainer
            self.components["rollouter"] = rollouter
            ray.get(rollouter.set_max_required_samples.remote())
            print("[ASYNC MAIN] Parallel trainer + rollouter init complete")

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _build_rollouter(self, config):
        """Construct the ``FullyAsyncRollouter`` actor handle, without initializing its workers.

        Single place that instantiates the rollouter actor, shared by both init paths.
        The parallel path (``use_trainer_do_validate=False``) calls this directly so it
        can overlap the rollouter's ``init_workers()`` with the trainer's;
        ``_create_rollouter`` wraps this with hybrid-WG injection + ``init_workers()``
        for the serial path.

        Constructing the handle deliberately does not inject a hybrid worker group: on
        the parallel path there is no trainer-side validation and hence no hybrid
        replicas. The serial path injects the hybrid WG in ``_create_rollouter`` after
        this call.
        """
        return FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

    def _create_rollouter(self, config) -> None:
        """Build the rollouter and fully initialize it (serial path).

        Wraps ``_build_rollouter`` with the steps required when the rollouter is not
        initialized in parallel with the trainer: inject the trainer's hybrid worker
        group (if present, for trainer-side validation) before ``init_workers()``, then
        initialize workers and register the actor in ``self.components``. Used on the
        ``use_trainer_do_validate=True`` path; the parallel path uses ``_build_rollouter``
        directly and initializes workers itself.
        """
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = self._build_rollouter(config)

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _build_trainer(self, config):
        """Construct the ``FullyAsyncTrainer`` actor handle, without initializing its workers.

        Single place that instantiates the trainer actor, shared by both init paths.
        The parallel path (``use_trainer_do_validate=False``) calls this directly so it
        can overlap the trainer's ``init_workers()`` with the rollouter's;
        ``_create_trainer`` wraps this with ``init_workers()`` for the serial path.
        """
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        return FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

    def _create_trainer(self, config) -> None:
        """Build the trainer and fully initialize it (serial path).

        Wraps ``_build_trainer`` with a blocking ``init_workers()`` and registers the
        actor in ``self.components``. Used on the ``use_trainer_do_validate=True`` path;
        the parallel path uses ``_build_trainer`` directly and initializes workers itself.
        """
        print("[ASYNC MAIN] Starting create trainer...")
        trainer = self._build_trainer(config)
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
