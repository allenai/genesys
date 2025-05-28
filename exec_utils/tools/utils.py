import docker
import io
import os
import re
import tarfile
import logging

from typing import Optional, Dict, Any
from time import sleep
from docker.client import DockerClient
from docker.models.images import Image as DockerImage
from docker.models.containers import Container

from ..aliases import ConfigType

util_logger = logging.getLogger('exec_utils.tools.utils')

__all__ = [
    "init_container",
    "copy_to_container",
    "copy_contents_to_container",
    "setup_environment"
]

def start_container(
        client: DockerClient,
        img_name: str,
        ctr_name: str
    ) -> Container:
    """Starts a Docker container 

    :param client: 
        The Docker client. 
    :param img_name: 
        The name of the Docker image. 
    :param ctr_name: 
        The name of the Docker container. 

    """

    try: 
        image = client.images.get(img_name)
    except docker.errors.ImageNotFound:
        image = client.images.pull(img_name)

    container = client.containers.run(
        image=image,
        name=ctr_name,
        detach=True,
        tty=True
    )

    return container

def wait_for_container(
        container: Container,
        timeout: Optional[int] = 120,
        stop_time: Optional[int] = 3
    ) -> Container:
    """Waits until the provided container is up and running 

    :param container: 
        The target container to load 
    :param timeout: 
        The timeout allowed for trying to reload 
    :param stop_time: 
        The time to stop between attempts. 
    :raises: 
        TimeoutError 

    """
    elapsed_time = 0

    while elapsed_time < timeout:
        sleep(stop_time)
        container.reload()
        if container.status == "running":
            return

        elapsed_time += stop_time

    raise TimeoutError(
        f"Container {container.name} did not start within {timeout} seconds"
    )

    
def init_container(client: DockerClient,config: ConfigType):
    """Initializes the target container 

    :param client: 
        The docker client 
    :param config: 
        The global configuration that contains details of 
        the container and image name, among other details.  
    
    """
    reset = config.reset_container_if_exists
    
    all_containers = [
        container.name for container in client.containers.list(all=True)
    ]

    ### find image 
    if config.ctr_name in all_containers and not reset:
        container = client.containers.get(config.ctr_name)
        wait_for_container(
            container,
            timeout=config.docker_timeout,
            stop_time=3
        )
        return container
        
    elif config.ctr_name in all_containers:  
        container = client.containers.get(config.ctr_name)
        container.stop()
        container.remove()
        
    container = start_container(
        client,
        config.img_name,
        config.ctr_name
    )        
    wait_for_container(
        container,
        timeout=config.docker_timeout,
        stop_time=3
    )
    
    return container

def copy_to_container(
        container: Container,
        src: str,
        dst_dir: str
    ) -> None :
    """Copies items to the container
    
    :param container: 
        The running container. 
    :param src: 
        The source item to copy (must be an absolute path) 
    :param dst_dir: 
        The destination inside of the the container. 
    """
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w|') as tar, open(src, 'rb') as f:
        info = tar.gettarinfo(fileobj=f)
        info.name = os.path.basename(src)
        tar.addfile(info, f)

    container.put_archive(dst_dir, stream.getvalue())

def copy_contents_to_container(
        container: Container,
        contents: str,
        dst_dir: str,
        dst_file_name: str = "file"
    ) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w|') as tar:
        tarinfo = tarfile.TarInfo(name=dst_file_name)
        tarinfo.size = len(contents)
        tar.addfile(tarinfo, io.BytesIO(contents.encode()))

    container.put_archive(dst_dir, stream.getvalue())


def setup_environment(container,config):
    """Run initial set up of the container 

    :param container: 
        The target container. 
    :param config: 
        The global configuration. 

    """
    container.exec_run(f"mkdir -p {config.executor_wdir}")
    util_logger.info(
        f"Creating executor wdir={config.executor_wdir} (if it doesnt exist)"
    )
    if not config.run_simple_execution:
        output_result = container.exec_run(
            "python /app/server.py",
            workdir=config.executor_wdir,
            detach=True
        )
        if output_result.exit_code is not None:
            raise ValueError(
                f"Failed to start Python server: {output_result}"
            )
        util_logger.info(f"Started Python server: output={output_result}")
    else:
        util_logger.info('Started simplified execution')
    
    #if config.copy_files:
    #    for source_file in os.listdir(config.copy_files):
            
