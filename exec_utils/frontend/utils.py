from abc import abstractmethod  
from types import ModuleType
from typing import (
    List,
    Union,
    Optional
)

from ..base import (
    UtilResource,
    UtilityModel
)

__all__ = [
    "SystemOut",
]

class SystemOut(UtilityModel):
    """Helper class for formatting system output to 
    our frontend code. 

    """
    print_output: list = []
    execution_graph: list = []
    formatted_output: list = []
    raw_data: dict = {}
    final_message: str = ""
    query: str = ""
    
    def download_func(self) -> None :
        """Specifies how to download output

        """
        
        for item in self.formatted_output:
            for message in item:
                yield message
                
    @property
    def markdown_string(self):
        output = f"### Query: {self.query}\n\n"
        for item in self.formatted_output:
            string_rep = '\n'.join(item)
            output += string_rep
            output += "\n"
        return output    

    def print_to_stream(
            self,
            stream: ModuleType,
            status: Optional[bool] = True,
        ) -> None:
        """Prints formatted output to stream """
        for k,message in enumerate(self.formatted_output):
            if status: 
                with stream.status(f"Running step: {k+1}"): 
                    for item in message: 
                        stream.write(
                            item,unsafe_allow_html=True
                        )
            else:
                for item in message: 
                    stream.write(
                        item,unsafe_allow_html=True
                    )
                
    def add_formatted_output(
            self,
            new_input: List[str],
            *,
            stream: Optional[ModuleType] = None,
        ):
        """Add formatted output to the system output 

        :param new_input: 
            New input to add to the output. 
        :param stream: 
            Streamlit module that can be written to. 

        """
        self.formatted_output.append(new_input)

        if stream: 
            for item in new_input: 
                stream.write(item,unsafe_allow_html=True)
