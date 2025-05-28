import docker
import io
import os
import re
import tarfile

from pathlib import Path
from typing import Optional, Dict, Any
from docker.client import DockerClient
from docker.models.images import Image as DockerImage
from docker.models.containers import Container

from ..utils import make_wdir 
from ..cache import setup_cache
from ..register import Registry
from ..aliases import ConfigType
from ..param import (
    ModuleParams,
    ParamField,
)

from .tool import BaseTool
from .utils import (
    init_container,
    copy_to_container,
    copy_contents_to_container,
    setup_environment,
)

__all__ = [
    "DockerExecutor",
]

@Registry(
    resource_type="tool_type",
    name="code_executor",
    #cache="query",
    cache=None,
)
class DockerExecutor(BaseTool):
    """A tool for executing code in Docker. 

    :param container 
        The Docker container that is used to execute commands. 
    """
    def __init__(
        self,
        client: DockerClient,
        container: DockerImage,
        config: ConfigType,
    ) -> None:
        """Creates `DockerExecutor` instance. 

        :param config: 
            The global configuration object.  
        """
        self.container = container
        self._client = client 
        self._config = config

        self._preprocess = None
        if self._config.execute_preprocess_func:
            self._preprocess = Registry.find_function(
                config.execute_preprocess_func
            )
            self.logging.info(f'Found preprocess function: {self._config.execute_preprocess_func}, {self._preprocess}')
            
    @property
    def name(self) -> str:
        """Returns the name of the tool"""
        return "docker_executor"

    def query(self,query: str,**kwargs) -> Dict[str,Any]:
        """The main point of execution for the code execution 
        object. 
        
        :param query: 
            The input query to the executor. 
        """
        action_type = kwargs.get("action_type","unknown")

        if self._preprocess:
            query = self._preprocess(query)
        
        if action_type == "execute":
            return self.run_command(query)

        elif action_type == "python":
            return self.run_python(query)
        
        else:
            self.logging.warning(
                f"Invalid action detected: {action_type}"
            )
            return 
            
    def run_python(self,code):
        """Executes python 

        """
        copy_contents_to_container(
            self.container,
            code,
            "/app/",
            dst_file_name="python_script.py"
        )
        if not self._config.run_simple_execution:
            output = self._run_command_bash(
                f"python /app/client.py /app/python_script.py"
            )
            self._run_command_bash(f"rm /app/python_script.py")
        else:
            output = self._run_command_bash(
                f"python -W ignore /app/python_script.py"
            )
            
        return output

    def run_command(self,command: str):
        """Runs a bash command in the container 

        :param command: 
            The full string rendering of the target bash
            command with possible `&&` delimiters        
        """
        output = ""
        commands = command.split("&&")

        for command in commands:

            if command.strip().startswith("cd "):

                relative_path = command.strip().split(" ")[1]
                absolute_path = self._run_command_bash(f"cd {relative_path} && pwd").strip()

                if "No such file or directory" in absolute_path:
                    output += f"cd: {relative_path}: No such file or directory\n"
                    break
                self._config.executor_wdir = absolute_path
                output += "\n"

            else:
                output += self._run_command_bash(command) + "\n"
                
        return output.rstrip()

    def _run_command_bash(self, command) -> str:
        """Executes a bash command `command`

        :param command: 
            The input bash command to run.
        """
        command = f'/bin/bash -c "{command}"'

        container_out = self.container.exec_run(
            command,
            workdir=self._config.executor_wdir
        )
        
        return container_out.output.decode("utf-8")
    
    
    @classmethod
    def from_config(cls,config,**kwargs):
        """Load executor instance from configuration 

        :param config: 
            The global configuration from which to build an 
            `DockerExecutor` instance. 
        :raises: 
            ValueError
        """
        if not config.executor_wdir:
            raise ValueError(
                "Must specify a working directory for the executor"
            )

        client = docker.from_env()
        container = init_container(client,config)

        setup_environment(container,config) 
        
        return cls(
            client,
            container,
            config
        )


@Registry(
    resource_type="tool_type",
    name="cached_code_executor",
    cache="query",
)
class CachedDockerExecutor(DockerExecutor):
    """A tool for executing code in Docker that includes caching. 

    :param container 
        The Docker container that is used to execute commands. 
    """
    
@Registry(
    resource_type="config",
    name="lm_exec_utils.tools.executor"
)
class Params(ModuleParams):
    """Parameters for model classes

    :param executor_wdir: 
        The working directory for the executor. 
    :param ctr_name: 
        The name of the docker container for running the executor. 
    :param img_name: 
        The name of the docker image for running the executor.
    :param docker_timeout: 
        The time limit for getting an image and container running. 
        
    """
    executor_wdir: str = ParamField(
        default='',
        metadata={
            "help" : 'The working directory for the executor',
        }
    )
    ctr_name: str = ParamField(
        default='nora-sandbox',
        metadata={"help" : 'The name of the docker container for the executor'}
    )
    img_name: str = ParamField(
        default='nora-image',
        metadata={
            "help" : 'The name of the docker image for the executor'
        }
    )
    reset_container_if_exists: bool = ParamField(
        default=False,
        metadata={
            "help" : "Reset an existing container if it already exists and is running",
            "exclude_hash" : True,
        }
    )
    run_simple_execution: bool = ParamField(
        default=False,
        metadata={
            "help" : "Don't run code in notebook style, just execute single scripts"
        }
    )
    docker_timeout: int = ParamField(
        default=120,
        metadata={
            "help" : "The timeout for getting an image running",
            #"exclude_hash" : True,
        }
    )
    copy_files: str = ParamField(
        default='',
        metadata={
            "help" : "Files in a directory to copy over to execution environment",
            "exclude_hash" : True,
        }
    )
    execute_preprocess_func: str = ParamField(
        default='',
        metadata={
            "help" : "A function to use for pre-processing a piece of code",
        }
    )
